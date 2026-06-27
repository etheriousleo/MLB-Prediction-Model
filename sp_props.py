"""
MLB Starting Pitcher Props — Strikeout & Performance Model
-----------------------------------------------------------
Install dependencies:
    pip install MLB-StatsAPI streamlit plotly numpy

Run:
    streamlit run sp_props.py
"""

import datetime
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import plotly.graph_objects as go
import statsapi
import streamlit as st

# Optional timezone + Streamlit threading context (both degrade gracefully if absent)
try:
    from zoneinfo import ZoneInfo
    _UTC = ZoneInfo("UTC")
    _ET  = ZoneInfo("America/New_York")
except Exception:
    _UTC = _ET = None

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:
    add_script_run_ctx = None
    def get_script_run_ctx():
        return None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SP Props", page_icon="⚾", layout="wide")

st.markdown("""
<style>
    .section-head {
        font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
        color: #888; margin: 1.5rem 0 0.5rem;
    }
    .prop-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .over  { color: #00c07a; font-weight: 700; }
    .under { color: #ff5252; font-weight: 700; }
    .push  { color: #f5c842; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

SEASON = datetime.datetime.now().year

# ── League-average baselines ───────────────────────────────────────────────────
# Used to contextualise pitcher and lineup stats
LEAGUE_K9       = 8.7    # average K/9 for starters
LEAGUE_BB9      = 3.1
LEAGUE_ERA      = 4.20
LEAGUE_WHIP     = 1.28
LEAGUE_IP_START = 5.2    # average innings per start
LEAGUE_K_PCT    = 0.225  # average batter strikeout rate (22.5%)
LEAGUE_BB_PCT   = 0.085  # average batter walk rate
LEAGUE_OPS      = 0.720  # league-average OPS baseline for the opponent run adjustment

# ── Model coefficients (hand-set defaults; tune against the backtest, not by eye) ──
OPP_K_SENSITIVITY = 2.5   # K/9 multiplier per unit of opponent K% deviation from league
FIP_WEIGHT        = 0.5   # FIP vs ERA split in the run-rate baseline (0=all ERA, 1=all FIP)
OPS_TO_RUNS       = 3.0   # opponent OPS deviation → ER/9 adjustment
FIP_CONSTANT      = 3.10  # league-normalizing constant paired with the 13/3/2 FIP weights

# Typical sportsbook K prop lines for reference
# Model will project K total and compare to these common lines
COMMON_K_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

# ── Ballpark factors (run suppression context) ─────────────────────────────────
PARK_RUN_FACTOR = {
    "COL": 1.18, "CIN": 1.07, "TEX": 1.06, "BOS": 1.05,
    "CHC": 1.04, "BAL": 1.04, "PHI": 1.03, "MIL": 1.03,
    "ARI": 1.03, "NYY": 1.02, "ATL": 1.02, "MIN": 1.01,
    "DET": 1.01, "HOU": 1.01, "LAD": 1.00, "SFG": 1.00,
    "WSN": 1.00, "STL": 1.00, "PIT": 0.99, "TOR": 0.99,
    "NYM": 0.99, "KCR": 0.99, "CLE": 0.99, "CHW": 0.98,
    "SEA": 0.97, "LAA": 0.97, "OAK": 0.97, "SDP": 0.96,
    "MIA": 0.96, "TBR": 0.95,
}

# ── Team lookup ────────────────────────────────────────────────────────────────
ABB_TO_FULL = {
    "ARI": "Arizona Diamondbacks",    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",       "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",            "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",         "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",        "DET": "Detroit Tigers",
    "HOU": "Houston Astros",          "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",      "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",           "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",         "NYM": "New York Mets",
    "NYY": "New York Yankees",        "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",   "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",        "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",        "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",          "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",       "WSN": "Washington Nationals",
}
FULL_TO_ABB = {v: k for k, v in ABB_TO_FULL.items()}

STATSAPI_NAME_MAP = {
    # Only non-identity remaps are needed; api_name_to_abb falls back to the raw name.
    # The Athletics' API name shifts with the franchise relocation, so alias the
    # variants back to the abbreviation key used in ABB_TO_FULL ("Oakland Athletics").
    "Athletics":            "Oakland Athletics",
    "Sacramento Athletics": "Oakland Athletics",
}

def api_name_to_abb(name: str):
    return FULL_TO_ABB.get(STATSAPI_NAME_MAP.get(name, name))


def ip_to_decimal(ip_val) -> float:
    """
    Convert MLB innings-pitched baseball notation to decimal innings.
    The API returns values like 6.1 (meaning 6⅓ innings) and 6.2 (6⅔),
    NOT true decimals. Outs are base-3, so .1 = 1/3 and .2 = 2/3.
    """
    ip_str = str(ip_val)
    parts  = ip_str.split(".")
    whole  = int(parts[0])
    outs   = int(parts[1]) if len(parts) > 1 else 0
    return whole + outs / 3


def compute_pitching_line(so: int, bb: int, hr: int, er: int, h: int,
                          ip: float, gs: int) -> dict:
    """
    Derive a pitcher's rate line from counting stats. Single source of truth for the
    FIP formula/constant and the season-to-date rates, shared by the live season fetch
    and the backtest's look-ahead-clean reconstruction so the two can never drift.
    Returns {} when there are no innings (caller treats that as "not enough data").
    """
    if ip <= 0:
        return {}
    return {
        "k9":   round(so * 9 / ip, 2),
        "bb9":  round(bb * 9 / ip, 2),
        "era":  round(er * 9 / ip, 2),
        "whip": round((h + bb) / ip, 3),
        "fip":  round((13 * hr + 3 * bb - 2 * so) / ip + FIP_CONSTANT, 2),
        "hr9":  round(hr * 9 / ip, 2),
        "ip":   round(ip, 1),
        "gs":   gs,
        "so":   so, "bb": bb, "hr": hr,
        "ip_per_start": round(ip / gs, 2) if gs > 0 else 0,
        "k_per_start":  round(so / gs, 1) if gs > 0 else 0,
    }


# ── Data fetching ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_todays_games() -> tuple[list, str]:
    today = datetime.datetime.today().strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(date=today, sportId=1)
        return [g for g in games if g.get("game_type", "") == "R"], ""
    except Exception as e:
        return [], str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_season_stats(player_id: int, season: int) -> dict:
    """Full season stats for a pitcher. The rate line is derived from counting stats via
    compute_pitching_line (same definition the backtest uses); only W-L is taken straight
    from the API, since it can't be reconstructed from a pitching log."""
    try:
        raw = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=pitching,type=season,season={season})",
        })
        people = raw.get("people", [])
        if not people:
            return {}
        splits = people[0].get("stats", [{}])[0].get("splits", [])
        s = splits[0].get("stat", {}) if splits else {}
        if not s:
            return {}
        # API returns innings in baseball notation (6.1 = 6⅓), not true decimal
        line = compute_pitching_line(
            so=int(s.get("strikeOuts", 0)),
            bb=int(s.get("baseOnBalls", 0)),
            hr=int(s.get("homeRuns", 0)),
            er=int(s.get("earnedRuns", 0)),
            h=int(s.get("hits", 0)),
            ip=ip_to_decimal(s.get("inningsPitched", "0") or 0),
            gs=int(s.get("gamesStarted", 0)),
        )
        if not line:
            return {}
        line["wins"]   = int(s.get("wins", 0))
        line["losses"] = int(s.get("losses", 0))
        return line
    except Exception:
        return {}


def last_n_starts_from_log(log: list[dict], n: int = 5) -> list[dict]:
    """
    Last N starts (most recent first) sliced from a full game log produced by
    fetch_pitcher_game_log_full. Pure helper, no API call — the live card and the
    backtest now share a single game-log fetch instead of pulling it twice.
    """
    starts = sorted((g for g in log if g["gs"] == 1),
                    key=lambda g: g["date"], reverse=True)[:n]
    return [
        {"date": g["date"], "opponent": g["opp_name"] or "?", "ip": g["ip"],
         "k": g["so"], "er": g["er"], "h": g["h"], "bb": g["bb"], "hr": g["hr"]}
        for g in starts
    ]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_batting(team_id: int, season: int) -> dict:
    """Fetch team batting stats — focused on K% and BB% for matchup analysis."""
    try:
        raw    = statsapi.get("team_stats", {
            "teamId": team_id, "stats": "season",
            "group": "hitting", "season": season,
        })
        splits = raw.get("stats", [{}])[0].get("splits", [])
        s      = splits[0].get("stat", {}) if splits else {}
        if not s:
            return {}
        pa  = max(float(s.get("plateAppearances", 0) or 0), 1)
        so  = float(s.get("strikeOuts",  0) or 0)
        bb  = float(s.get("baseOnBalls", 0) or 0)
        return {
            "k_pct":  round(so / pa, 3),
            "bb_pct": round(bb / pa, 3),
            "obp":    float(s.get("obp", 0) or 0),
            "slg":    float(s.get("slg", 0) or 0),
            "ops":    float(s.get("ops", 0) or 0),
            "avg":    float(s.get("avg", 0) or 0),
            "runs_pg": round(float(s.get("runs", 0) or 0) /
                             max(int(s.get("gamesPlayed", 1) or 1), 1), 2),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_by_id(player_id: int, season: int) -> tuple[dict, list]:
    """(season_stats, last_5_starts) — both built from one shared game-log fetch."""
    log = fetch_pitcher_game_log_full(player_id, season)
    return fetch_pitcher_season_stats(player_id, season), last_n_starts_from_log(log)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_id_by_name(name: str) -> int | None:
    try:
        res = statsapi.lookup_player(name)
        return res[0]["id"] if res else None
    except Exception:
        return None


# ── Snapshot system (opponent K% as-of-date, for look-ahead-clean backtesting) ──
# A pitcher's season-to-date line, recent form, AND the actual result of each start
# are all reconstructed from the immutable game log at backtest time — so they need
# NO snapshot and have full historical coverage from day one. The one input that
# can't come from a pitcher's own log is the OPPONENT lineup's K% as it stood before
# the game (team K% drifts over a season). We snapshot team batting daily so the
# backtest can read the opponent's pre-game K%; until snapshots accumulate it falls
# back to current-season K% (a small, stable bias, reported as coverage). Same
# snapshot idea as mlb_app, scoped to the one quantity that actually needs it.
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots_spprops")

def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def team_snapshot_path(date_str: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"teambat_{date_str}.json")

def save_team_batting_snapshot(team_batting: dict, date_str: str = None):
    """Persist {abb: batting_dict} for today so backtests can read pre-game opp K%."""
    ensure_snapshot_dir()
    if date_str is None:
        date_str = datetime.datetime.today().strftime("%Y-%m-%d")
    clean = {k: v for k, v in (team_batting or {}).items() if k and v}
    if not clean:
        return
    path = team_snapshot_path(date_str)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
            existing.update(clean)   # merge — never overwrite a fuller snapshot with a sparse one
            clean = existing
        except Exception:
            pass
    try:
        with open(path, "w") as f:
            json.dump(clean, f)
    except Exception:
        pass  # best-effort

def list_team_snapshots() -> list[str]:
    ensure_snapshot_dir()
    return sorted(
        f.replace("teambat_", "").replace(".json", "")
        for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("teambat_") and f.endswith(".json")
    )

def load_all_team_snapshots() -> dict:
    """Return {date_str: {abb: batting}} for every saved team-batting snapshot."""
    out = {}
    for d in list_team_snapshots():
        try:
            with open(team_snapshot_path(d), "r") as f:
                out[d] = json.load(f)
        except Exception:
            continue
    return out

def opp_batting_before(date_str: str, opp_abb: str, snaps: dict, current: dict):
    """
    Opponent team batting as of just before date_str. Uses the most recent team
    snapshot strictly before the game date; falls back to current-season batting if
    none exists. Returns (batting_dict, used_snapshot_bool).
    """
    try:
        game_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return current.get(opp_abb, {}), False
    for snap_date in sorted(snaps.keys(), reverse=True):
        try:
            if datetime.datetime.strptime(snap_date, "%Y-%m-%d") < game_dt:
                bat = snaps[snap_date].get(opp_abb)
                if bat:
                    return bat, True
        except ValueError:
            continue
    return current.get(opp_abb, {}), False


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_game_log_full(player_id: int, season: int) -> list[dict]:
    """
    Full season game log (all appearances), ascending by date. Each entry carries the
    per-game counting stats needed to rebuild season-to-date lines and to score the
    actual outcome of each start.
    """
    try:
        raw = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=pitching,type=gameLog,season={season})",
        })
        people = raw.get("people", [])
        if not people:
            return []
        splits = people[0].get("stats", [{}])[0].get("splits", [])
        log = []
        for sp in splits:
            s = sp.get("stat", {})
            log.append({
                "date":     sp.get("date", "")[:10],
                "opp_name": sp.get("opponent", {}).get("name", ""),
                "is_home":  sp.get("isHome", None),   # used only for park (→ proj_er); absent is fine
                "gs":       int(s.get("gamesStarted", 0) or 0),
                "so":       int(s.get("strikeOuts", 0) or 0),
                "bb":       int(s.get("baseOnBalls", 0) or 0),
                "hr":       int(s.get("homeRuns", 0) or 0),
                "er":       int(s.get("earnedRuns", 0) or 0),
                "h":        int(s.get("hits", 0) or 0),
                "ip":       ip_to_decimal(s.get("inningsPitched", 0) or 0),
            })
        log.sort(key=lambda x: x["date"])
        return log
    except Exception:
        return []


def reconstruct_state_before(log: list[dict], cutoff_date: str):
    """
    Rebuild a pitcher's season-to-date stat line and last-5 starts using ONLY
    appearances strictly before cutoff_date. Returns (season_stats, recent_starts) or
    (None, None) if there isn't enough prior data. No look-ahead: every input predates
    the game being projected. Keys mirror fetch_pitcher_season_stats so the same
    project_strikeouts() runs unchanged.
    """
    prior = [g for g in log if g["date"] and g["date"] < cutoff_date]
    if not prior:
        return None, None
    so = sum(g["so"] for g in prior); bb = sum(g["bb"] for g in prior)
    hr = sum(g["hr"] for g in prior); er = sum(g["er"] for g in prior)
    h  = sum(g["h"]  for g in prior); ip = sum(g["ip"] for g in prior)
    gs = sum(g["gs"] for g in prior)
    season_stats = compute_pitching_line(so=so, bb=bb, hr=hr, er=er, h=h, ip=ip, gs=gs)
    if not season_stats:
        return None, None
    prior_starts  = [g for g in prior if g["gs"] == 1]
    recent_starts = [
        {"ip": g["ip"], "k": g["so"], "er": g["er"],
         "date": g["date"], "opponent": g["opp_name"]}
        for g in prior_starts[-5:]
    ]
    return season_stats, recent_starts


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_team_batting(season: int) -> dict:
    """Current-season batting for all 30 teams, keyed by abbreviation (backtest fallback)."""
    out = {}
    try:
        teams_resp = statsapi.get("teams", {"sportId": 1, "season": season})
        for t in teams_resp.get("teams", []):
            if t.get("sport", {}).get("id") != 1:
                continue
            abb = api_name_to_abb(t.get("name", ""))
            if not abb:
                continue
            bat = fetch_team_batting(t["id"], season)
            if bat:
                out[abb] = bat
    except Exception:
        pass
    return out


# ── Model ──────────────────────────────────────────────────────────────────────
def project_strikeouts(pitcher: dict, recent_starts: list, opp_batting: dict,
                       park_abb: str, n_recent_weight: float = 0.4) -> dict:
    """
    Project strikeouts for a starting pitcher in today's game.

    Method:
    1. Blend season K/9 with recent-form K/9 (weighted average of last 5 starts)
    2. Adjust for opposing lineup K% vs league average
    3. Estimate expected innings pitched
    4. Project K = adjusted_K9 * (expected_IP / 9)
    5. Snap to 0.5 increments

    Returns a dict with all projection components.
    """
    if not pitcher:
        return {}

    season_k9        = pitcher.get("k9",          LEAGUE_K9)
    season_ip_start  = pitcher.get("ip_per_start", LEAGUE_IP_START)
    season_k_per_start = pitcher.get("k_per_start", 0)

    # Recent form K/9 from last N starts
    if recent_starts:
        recent_ip = sum(s["ip"] for s in recent_starts)
        recent_k  = sum(s["k"]  for s in recent_starts)
        recent_k9 = round((recent_k / recent_ip * 9), 2) if recent_ip > 0 else season_k9
        recent_ip_avg = round(recent_ip / len(recent_starts), 2)
        recent_er_avg = round(sum(s["er"] for s in recent_starts) / len(recent_starts), 2)
        recent_k_avg  = round(recent_k / len(recent_starts), 1)
    else:
        recent_k9     = season_k9
        recent_ip_avg = season_ip_start
        recent_er_avg = 0.0  # no recent starts; safe default for any downstream rendering
        recent_k_avg  = season_k_per_start

    # Blend season and recent K/9
    blended_k9 = round(
        season_k9 * (1 - n_recent_weight) + recent_k9 * n_recent_weight, 2
    )

    # Opposing lineup K% adjustment
    # If opponent K% is 5pp above league avg (0.275 vs 0.225), pitcher gets a boost
    opp_k_pct    = opp_batting.get("k_pct", LEAGUE_K_PCT)
    k_pct_delta  = opp_k_pct - LEAGUE_K_PCT          # positive = more Ks for pitcher
    k9_adj       = round(blended_k9 * (1 + k_pct_delta * OPP_K_SENSITIVITY), 2)

    # Expected innings — blend season avg with recent, cap at 7.0
    exp_ip = min(7.0, round(
        season_ip_start * (1 - n_recent_weight) + recent_ip_avg * n_recent_weight, 1
    ))

    # Projected Ks
    proj_k_raw = k9_adj * (exp_ip / 9)
    proj_k     = round(round(proj_k_raw * 2) / 2, 1)  # snap to 0.5

    # Projected runs allowed
    opp_ops      = opp_batting.get("ops", LEAGUE_OPS)
    park_factor  = PARK_RUN_FACTOR.get(park_abb, 1.00)

    # Expected ER per 9 innings, adjusted for park and opponent.
    # Baseline blends FIP and ERA: FIP strips out defense/sequencing luck and is
    # more predictive of future runs (which is why the card flags it as the more
    # predictive stat), while ERA still carries real run-prevention signal.
    # Split is the module-level FIP_WEIGHT (0..1).
    pitcher_era = pitcher.get("era", LEAGUE_ERA)
    pitcher_fip = pitcher.get("fip", pitcher_era)
    run_rate    = pitcher_fip * FIP_WEIGHT + pitcher_era * (1 - FIP_WEIGHT)
    opp_ops_adj = (opp_ops - LEAGUE_OPS) * OPS_TO_RUNS    # ops deviation → run adjustment
    # Park scales the whole expected run environment, opponent offense included, so a
    # strong lineup in a hitter's park compounds rather than adding a flat bump.
    era_adj     = (run_rate + opp_ops_adj) * park_factor
    proj_er_raw = era_adj * (exp_ip / 9)
    proj_er     = round(round(proj_er_raw * 2) / 2, 1)

    # Confidence in the projection
    gs = pitcher.get("gs", 0)
    if gs >= 10 and recent_starts and len(recent_starts) >= 3:
        conf = "High"
    elif gs >= 5:
        conf = "Moderate"
    else:
        conf = "Low — small sample"

    # K prop recommendations vs common lines
    k_props = {}
    for line in COMMON_K_LINES:
        edge = round(proj_k - line, 1)
        if edge >= 1.0:
            rec = ("OVER", "#00c07a", f"proj {proj_k} vs line {line} (+{edge})")
        elif edge <= -1.0:
            rec = ("UNDER", "#ff5252", f"proj {proj_k} vs line {line} ({edge})")
        else:
            rec = ("PUSH", "#f5c842", f"proj {proj_k} vs line {line} (edge too small)")
        k_props[line] = rec

    return {
        "season_k9":      season_k9,
        "recent_k9":      recent_k9,
        "blended_k9":     blended_k9,
        "opp_k_pct":      opp_k_pct,
        "k9_adj":         k9_adj,
        "exp_ip":         exp_ip,
        "proj_k":         proj_k,
        "proj_er":        proj_er,
        "recent_k_avg":   recent_k_avg,
        "recent_er_avg":  recent_er_avg,
        "recent_ip_avg":  recent_ip_avg,
        "park_factor":    park_factor,
        "confidence":     conf,
        "k_props":        k_props,
        "gs":             gs,
    }


def quality_tier(k9: float) -> tuple[str, str]:
    """Return (label, color) for a pitcher's K/9 tier."""
    if k9 >= 10.5: return "Elite",        "#00c07a"
    if k9 >= 9.0:  return "Above avg",    "#5bc17a"
    if k9 >= 7.5:  return "Average",      "#f5c842"
    if k9 >= 6.0:  return "Below avg",    "#f5a623"
    return             "Contact pitcher", "#ff5252"


def _load_sp(task: dict):
    """Resolve a pitcher's id if needed, fetch their season line + last-5, and build the
    render row — or return None if they can't be loaded or fail the min-GS filter. Reads
    only caches and the task dict, so it's safe to run inside the thread pool."""
    pid = task["sp_pid"]
    if not isinstance(pid, int) or not pid:
        pid = fetch_pitcher_id_by_name(task["sp_name"])
    if not pid:
        return None
    season_stats, recent_starts = fetch_pitcher_by_id(pid, SEASON)
    if not season_stats or season_stats.get("gs", 0) < task["min_gs"]:
        return None
    proj = project_strikeouts(
        season_stats, recent_starts, task["opp_batting"],
        task["venue_abb"], task["recent_weight"],
    )
    if not proj:
        return None
    tier_label, tier_color = quality_tier(season_stats.get("k9", 0))
    return {
        "pid": pid, "name": task["sp_name"],
        "team": task["team_name"], "team_abb": task["team_abb"],
        "is_home": task["is_home"], "opp": task["opp_name"], "opp_abb": task["opp_abb"],
        "venue": task["venue_name"], "venue_abb": task["venue_abb"],
        "season_stats": season_stats, "recent_starts": recent_starts,
        "proj": proj, "tier_label": tier_label, "tier_color": tier_color,
        "game_time": task["game_time"],
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚾ SP Props Model")
    st.caption(f"Season: {SEASON} · statsapi")

    st.markdown("---")
    st.markdown("##### Model settings")
    recent_weight = st.slider(
        "Recent form weight", 0, 100, 40, step=10,
        help="How much weight to give the last 5 starts vs full-season stats."
    ) / 100

    st.markdown("---")
    st.markdown("##### What to show")
    show_game_log   = st.checkbox("Show last 5 starts", value=True)
    show_k_props    = st.checkbox("Show K prop breakdown", value=True)
    min_gs_filter   = st.slider("Min games started (filter out openers)", 1, 10, 3)

    st.markdown("---")
    st.caption("Projections use season K/9 blended with recent form, "
               "adjusted for opposing lineup K% and ballpark factor.")


# ── Main panel ─────────────────────────────────────────────────────────────────
_today = datetime.datetime.today()
today_label = _today.strftime("%A, %B ") + str(_today.day)  # str(day) never has leading zero
st.markdown(f"## ⚾ SP Props — {today_label}")
st.caption(
    "Projected strikeouts, runs allowed, and innings pitched for every "
    "starting pitcher on today's slate. Sorted by projected Ks."
)

with st.spinner("Fetching today's schedule..."):
    todays_games, err = fetch_todays_games()

if err:
    st.error(f"Could not load schedule: {err}")
    st.stop()

if not todays_games:
    st.info("No MLB regular season games today. Check back tomorrow!")
    st.stop()

# ── Build projections for every SP ────────────────────────────────────────────
# Two parallel fetch passes (team batting, then pitcher lines) replace the old
# per-game serial loop. Every fetch is cached and I/O-bound, so a small thread pool
# turns a cold full-slate load from dozens of serial round-trips into a few seconds.
all_pitchers = []
todays_team_batting = {}   # {abb: batting} collected across today's games → snapshotted
MAX_WORKERS = 8

# Carry the main thread's Streamlit context into workers so cached calls run cleanly
# (no "missing ScriptRunContext" log spam). Degrades to a no-op if the API isn't present.
_ctx = get_script_run_ctx()
def _run_with_ctx(fn, *args):
    if add_script_run_ctx is not None and _ctx is not None:
        add_script_run_ctx(threading.current_thread(), _ctx)
    return fn(*args)

# Pass 1 — fetch each team's batting once, in parallel.
team_abb_by_id = {}
for game in todays_games:
    if game.get("home_id"):
        team_abb_by_id[game["home_id"]] = api_name_to_abb(game.get("home_name", ""))
    if game.get("away_id"):
        team_abb_by_id[game["away_id"]] = api_name_to_abb(game.get("away_name", ""))

progress = st.progress(0.0, text="Loading team batting...")
team_batting_by_id = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futs = {pool.submit(_run_with_ctx, fetch_team_batting, tid, SEASON): tid
            for tid in team_abb_by_id}
    for done, fut in enumerate(as_completed(futs)):
        tid = futs[fut]
        try:
            bat = fut.result()
        except Exception:
            bat = {}
        team_batting_by_id[tid] = bat
        abb = team_abb_by_id.get(tid)
        if abb and bat:
            todays_team_batting[abb] = bat
        progress.progress((done + 1) / max(len(futs), 1), text="Loading team batting...")

# Build one task per startable SP, with its opponent's batting already resolved.
tasks = []
for game in todays_games:
    h_name = game.get("home_name", ""); a_name = game.get("away_name", "")
    h_abb  = api_name_to_abb(h_name);   a_abb  = api_name_to_abb(a_name)
    venue_abb  = h_abb                   # park is always the home team's
    venue_name = game.get("venue_name", "")
    h_bat = team_batting_by_id.get(game.get("home_id"), {})
    a_bat = team_batting_by_id.get(game.get("away_id"), {})
    for sp_name, sp_pid, is_home, opp_batting, opp_name, opp_abb in [
        (game.get("home_probable_pitcher", "TBD"), game.get("home_pitcher_id"),
         True,  a_bat, a_name, a_abb),
        (game.get("away_probable_pitcher", "TBD"), game.get("away_pitcher_id"),
         False, h_bat, h_name, h_abb),
    ]:
        if not sp_name or sp_name == "TBD":
            continue
        tasks.append({
            "sp_name": sp_name, "sp_pid": sp_pid, "is_home": is_home,
            "opp_batting": opp_batting, "opp_name": opp_name, "opp_abb": opp_abb,
            "team_name": h_name if is_home else a_name,
            "team_abb":  h_abb  if is_home else a_abb,
            "venue_abb": venue_abb, "venue_name": venue_name,
            "game_time": game.get("game_datetime", ""),
            "min_gs": min_gs_filter, "recent_weight": recent_weight,
        })

# Pass 2 — fetch every pitcher's line in parallel and assemble the render rows.
progress.progress(0.0, text="Loading pitcher data...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futs2 = [pool.submit(_run_with_ctx, _load_sp, t) for t in tasks]
    for done, fut in enumerate(as_completed(futs2)):
        try:
            row = fut.result()
        except Exception:
            row = None
        if row:
            all_pitchers.append(row)
        progress.progress((done + 1) / max(len(futs2), 1), text="Loading pitcher data...")

progress.empty()

# Save today's team batting so future backtests can read pre-game opponent K%.
save_team_batting_snapshot(todays_team_batting)

if not all_pitchers:
    st.info("No starting pitcher data available for today's games yet. "
            "Probable starters are usually posted a few hours before game time.")
    st.stop()

# Sort by projected Ks descending
all_pitchers.sort(key=lambda x: (-x["proj"].get("proj_k", 0), x["name"]))

st.caption(f"{len(all_pitchers)} starters with projections · "
           f"recent form weight: {round(recent_weight*100)}% · "
           f"min {min_gs_filter} GS filter")
st.markdown("")

# ── Render each pitcher card ───────────────────────────────────────────────────
for p in all_pitchers:
    proj         = p["proj"]
    stats        = p["season_stats"]
    tier_color   = p["tier_color"]
    conf         = proj["confidence"]
    conf_color   = "#00c07a" if "High" in conf else ("#f5c842" if "Moderate" in conf else "#888")

    # Game time — show US Eastern (falls back to UTC if tz data is unavailable)
    try:
        gt = datetime.datetime.strptime(p["game_time"], "%Y-%m-%dT%H:%M:%SZ")
        if _ET is not None:
            gt_et = gt.replace(tzinfo=_UTC).astimezone(_ET)
            time_str = gt_et.strftime("%I:%M %p ").lstrip("0") + gt_et.strftime("%Z")
        else:
            time_str = gt.strftime("%I:%M %p UTC").lstrip("0")
    except Exception:
        time_str = ""

    ha_str = "HOME" if p["is_home"] else "AWAY"

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid {tier_color}33;'
        f'border-left:4px solid {tier_color};border-radius:12px;'
        f'padding:18px 22px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            f'<span style="font-size:20px;font-weight:800;">{p["name"]}</span> '
            f'<span style="color:#555;font-size:13px;">({ha_str}) · {p["team"]} vs {p["opp"]}</span><br>'
            f'<span style="font-size:11px;color:#888;">{p["venue"]} · {time_str}</span>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div style="text-align:right;">'
            f'<span style="font-size:11px;color:{tier_color};font-weight:700;">{p["tier_label"]}</span><br>'
            f'<span style="font-size:11px;color:{conf_color};">Confidence: {conf}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Key projections ────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">Today\'s projections</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Proj strikeouts", f"{proj['proj_k']}")
    with c2:
        st.metric("Proj innings", f"{proj['exp_ip']}")
    with c3:
        st.metric("Proj runs allowed", f"{proj['proj_er']}")
    with c4:
        st.metric("Adj K/9", f"{proj['k9_adj']}",
                  help="K/9 adjusted for opposing lineup strikeout rate")
    with c5:
        opp_k_pct_pct = round(proj['opp_k_pct'] * 100, 1)
        delta_k = round((proj['opp_k_pct'] - LEAGUE_K_PCT) * 100, 1)
        st.metric("Opp K%", f"{opp_k_pct_pct}%",
                  delta=f"{'+' if delta_k>=0 else ''}{delta_k}pp vs avg",
                  help="Higher opp K% = more strikeouts expected for the pitcher")

    # ── Season stats ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">Season stats</p>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1: st.metric("ERA",  f"{stats.get('era', '—')}")
    with s2: st.metric("FIP",  f"{stats.get('fip', '—')}")
    with s3: st.metric("WHIP", f"{stats.get('whip', '—')}")
    with s4: st.metric("K/9",  f"{stats.get('k9', '—')}")
    with s5: st.metric("BB/9", f"{stats.get('bb9', '—')}")
    with s6: st.metric("GS",   f"{stats.get('gs', 0)} ({stats.get('wins',0)}W-{stats.get('losses',0)}L)")

    # Projection breakdown bar
    st.markdown('<p class="section-head">Projection breakdown</p>', unsafe_allow_html=True)
    fig = go.Figure()

    components = {
        f"Season K/9 ({proj['season_k9']})":   proj["season_k9"],
        f"Recent K/9 ({proj['recent_k9']})":    proj["recent_k9"],
        f"Blended K/9 ({proj['blended_k9']})":  proj["blended_k9"],
        f"Adj for opp ({proj['k9_adj']})":       proj["k9_adj"],
    }
    colors = ["#3d8bff", "#5bc17a", "#f5c842", tier_color]

    fig.add_trace(go.Bar(
        x=list(components.values()),
        y=list(components.keys()),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in components.values()],
        textposition="outside",
        textfont=dict(color="#ccc", size=11),
    ))
    fig.add_vline(x=LEAGUE_K9, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Lg avg", annotation_font_color="rgba(255,255,255,0.4)",
                  annotation_position="top")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=175, margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, max(max(components.values()) * 1.2, 12)],
                   tickfont=dict(color="#888", size=10),
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(tickfont=dict(color="#ccc", size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── K prop lines ───────────────────────────────────────────────────────────
    if show_k_props:
        st.markdown('<p class="section-head">K prop lines</p>', unsafe_allow_html=True)
        prop_cols = st.columns(len(COMMON_K_LINES))
        for j, line in enumerate(COMMON_K_LINES):
            rec, col, reason = proj["k_props"][line]
            with prop_cols[j]:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;'
                    f'padding:10px;text-align:center;border:1px solid {col}44;">'
                    f'<div style="font-size:11px;color:#888;margin-bottom:4px;">O/U {line}</div>'
                    f'<div style="font-size:15px;font-weight:800;color:{col};">{rec}</div>'
                    f'<div style="font-size:10px;color:#666;margin-top:4px;">{reason}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Last 5 starts game log ─────────────────────────────────────────────────
    if show_game_log and p["recent_starts"]:
        st.markdown('<p class="section-head">Last 5 starts</p>', unsafe_allow_html=True)
        log_cols = st.columns(len(p["recent_starts"]))
        for j, start in enumerate(p["recent_starts"]):
            k_color = "#00c07a" if start["k"] >= 6 else ("#f5c842" if start["k"] >= 4 else "#ff5252")
            with log_cols[j]:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;'
                    f'padding:10px;text-align:center;">'
                    f'<div style="font-size:10px;color:#666;">{start["date"]}</div>'
                    f'<div style="font-size:10px;color:#888;margin-bottom:6px;">vs {start["opponent"]}</div>'
                    f'<div style="font-size:18px;font-weight:800;color:{k_color};">{start["k"]}K</div>'
                    f'<div style="font-size:11px;color:#aaa;">{start["ip"]} IP · {start["er"]} ER</div>'
                    f'<div style="font-size:10px;color:#666;">{start["h"]}H · {start["bb"]}BB</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Trend line for last 5 starts Ks
        if len(p["recent_starts"]) >= 3:
            dates_rev = [s["date"] for s in reversed(p["recent_starts"])]
            ks_rev    = [s["k"]    for s in reversed(p["recent_starts"])]
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=dates_rev, y=ks_rev, mode="lines+markers+text",
                line=dict(color=tier_color, width=2),
                marker=dict(size=8, color=tier_color),
                text=[str(k) for k in ks_rev],
                textposition="top center",
                textfont=dict(color="#ccc", size=11),
                name="Strikeouts",
            ))
            fig_trend.add_hline(y=proj["proj_k"], line_dash="dash",
                                line_color="rgba(255,255,255,0.4)",
                                annotation_text=f"Proj: {proj['proj_k']}",
                                annotation_font_color="rgba(255,255,255,0.6)",
                                annotation_position="right")
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=160, margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(tickfont=dict(color="#888", size=10),
                           showgrid=False, zeroline=False),
                yaxis=dict(tickfont=dict(color="#888", size=10),
                           showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                           zeroline=False, title="Strikeouts"),
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")


# ── Summary table ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Quick Reference — All Starters Today")
st.caption("Sorted by projected strikeouts. Use this to quickly scan the full slate.")

summary_rows = []
for p in all_pitchers:
    proj  = p["proj"]
    stats = p["season_stats"]
    # Best-priced actionable line: the line nearest the projection that still clears the
    # edge. For overs that's the highest qualifying line; for unders, the lowest — both
    # sit just past the projection, where the odds are closest to even money.
    best_line, best_rec, best_col, best_reason = None, None, "#888", ""
    for line in reversed(COMMON_K_LINES):       # descending → highest qualifying OVER
        rec, col, reason = proj["k_props"][line]
        if rec == "OVER":
            best_line, best_rec, best_col, best_reason = line, "OVER", col, reason
            break
    if not best_line:
        for line in COMMON_K_LINES:             # ascending → lowest qualifying UNDER
            rec, col, reason = proj["k_props"][line]
            if rec == "UNDER":
                best_line, best_rec, best_col, best_reason = line, "UNDER", col, reason
                break

    summary_rows.append({
        "Pitcher":    p["name"],
        "Team":       p["team"],
        "Opponent":   p["opp"],
        "Venue":      p["venue"],
        "K/9":        stats.get("k9", "—"),
        "ERA":        stats.get("era", "—"),
        "Proj K":     proj["proj_k"],
        "Proj IP":    proj["exp_ip"],
        "Proj ER":    proj["proj_er"],
        "Opp K%":     f"{round(proj['opp_k_pct']*100,1)}%",
        "Best line":  f"O {best_line}" if best_rec == "OVER" else (f"U {best_line}" if best_rec else "—"),
        "Rec":        best_rec or "—",
        "_rec_color": best_col,
        "GS":         stats.get("gs", 0),
    })

# Render as a clean HTML table
header_cols = ["Pitcher", "vs", "Venue", "K/9", "ERA", "Proj K", "Proj IP", "Proj ER", "Opp K%", "Best line", "Rec"]
header_html = "".join(
    f"<th style='padding:6px 10px;color:#888;font-size:10px;letter-spacing:1.5px;"
    f"text-transform:uppercase;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1);'>{h}</th>"
    for h in header_cols
)

rows_html = ""
for r in summary_rows:
    rc = r["_rec_color"]
    rows_html += (
        f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
        f"<td style='padding:8px 10px;color:#ccc;font-weight:600;'>{r['Pitcher']}</td>"
        f"<td style='padding:8px 10px;color:#888;font-size:12px;'>{r['Opponent']}</td>"
        f"<td style='padding:8px 10px;color:#666;font-size:11px;'>{r['Venue']}</td>"
        f"<td style='padding:8px 10px;color:#ccc;'>{r['K/9']}</td>"
        f"<td style='padding:8px 10px;color:#ccc;'>{r['ERA']}</td>"
        f"<td style='padding:8px 10px;font-weight:700;color:#f5c842;font-size:15px;'>{r['Proj K']}</td>"
        f"<td style='padding:8px 10px;color:#ccc;'>{r['Proj IP']}</td>"
        f"<td style='padding:8px 10px;color:#ccc;'>{r['Proj ER']}</td>"
        f"<td style='padding:8px 10px;color:#aaa;'>{r['Opp K%']}</td>"
        f"<td style='padding:8px 10px;color:{rc};font-weight:600;'>{r['Best line']}</td>"
        f"<td style='padding:8px 10px;font-weight:800;color:{rc};'>{r['Rec']}</td>"
        f"</tr>"
    )

st.markdown(
    f"<div style='overflow-x:auto;'>"
    f"<table style='width:100%;border-collapse:collapse;'>"
    f"<thead><tr>{header_html}</tr></thead>"
    f"<tbody>{rows_html}</tbody>"
    f"</table></div>",
    unsafe_allow_html=True,
)

st.markdown("")
st.caption(
    "⚠️ Projections are model estimates based on season stats, recent form, "
    "opposing lineup K%, and ballpark factors. Always cross-reference with "
    "actual sportsbook lines before placing any bets. For entertainment purposes only."
)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST / VALIDATION  (snapshot + replay pattern, mirroring mlb_app)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔬 Model Validation — Backtest")
st.caption(
    "Replays every prior start this season for the pitchers on today's slate using ONLY "
    "data available before each start. Season line and last-5 form are rebuilt from the "
    "immutable game log (no look-ahead, full coverage); opponent K% comes from a pre-game "
    "team snapshot when one exists, else current-season K%. Measures projection error and "
    "the hit rate of the model's OVER/UNDER calls at the fixed reference lines — not "
    "against live sportsbook lines, which aren't stored. Park (and thus Proj ER) uses "
    "home/away when the log provides it."
)

bt_min_prior = st.slider(
    "Min prior starts required to evaluate a start", 1, 10, 3,
    help="Skip a start if the pitcher had fewer than this many starts before it — "
         "early-season projections off 1–2 starts are noise.")
run_bt = st.button("▶ Run backtest on today's starters")

if run_bt:
    with st.spinner("Loading current team batting (opponent-K% fallback)..."):
        current_team_bat = fetch_all_team_batting(SEASON)
    team_snaps = load_all_team_snapshots()

    k_errs, er_errs, ip_errs = [], [], []
    line_stats = {ln: {"bets": 0, "hits": 0} for ln in COMMON_K_LINES}
    tier_stats = {t: {"bets": 0, "hits": 0} for t in ["High", "Moderate", "Low"]}
    opp_clean = 0
    eval_rows = []
    pitchers_evaluated = 0

    prog = st.progress(0.0, text="Backtesting...")
    pool = [p for p in all_pitchers if p.get("pid")]
    for idx, p in enumerate(pool):
        prog.progress((idx + 1) / max(len(pool), 1), text=f"Backtesting {p['name']}...")
        try:
            log = fetch_pitcher_game_log_full(p["pid"], SEASON)
        except Exception:
            continue
        if not log:
            continue
        evaluated_any = False
        for g in [x for x in log if x["gs"] == 1]:
            cutoff = g["date"]
            if not cutoff:
                continue
            prior_starts = [x for x in log if x["gs"] == 1 and x["date"] and x["date"] < cutoff]
            if len(prior_starts) < bt_min_prior:
                continue
            season_stats, recent_starts = reconstruct_state_before(log, cutoff)
            if not season_stats:
                continue

            opp_abb = api_name_to_abb(g.get("opp_name", "")) or ""
            opp_bat, used_snap = opp_batting_before(cutoff, opp_abb, team_snaps, current_team_bat)
            opp_clean += 1 if used_snap else 0

            is_home = g.get("is_home")
            if is_home is True:
                park_abb = p.get("team_abb")
            elif is_home is False:
                park_abb = opp_abb
            else:
                park_abb = None   # unknown → neutral park (affects Proj ER only, not Proj K)

            proj = project_strikeouts(season_stats, recent_starts, opp_bat, park_abb, recent_weight)
            if not proj:
                continue

            actual_k  = g["so"]
            actual_er = g["er"]
            actual_ip = round(g["ip"], 1)
            k_errs.append(abs(proj["proj_k"]  - actual_k))
            er_errs.append(abs(proj["proj_er"] - actual_er))
            ip_errs.append(abs(proj["exp_ip"]  - actual_ip))

            conf = proj.get("confidence", "")
            tkey = "High" if "High" in conf else ("Moderate" if "Moderate" in conf else "Low")

            for ln in COMMON_K_LINES:
                rec = proj["k_props"][ln][0]
                if rec == "OVER":
                    hit = actual_k > ln
                elif rec == "UNDER":
                    hit = actual_k < ln
                else:
                    continue   # PUSH → model made no call
                line_stats[ln]["bets"] += 1
                line_stats[ln]["hits"] += 1 if hit else 0
                tier_stats[tkey]["bets"] += 1
                tier_stats[tkey]["hits"] += 1 if hit else 0

            eval_rows.append({
                "date": cutoff, "pitcher": p["name"], "opp": g.get("opp_name", ""),
                "proj_k": proj["proj_k"], "actual_k": actual_k, "tier": tkey,
            })
            evaluated_any = True
        if evaluated_any:
            pitchers_evaluated += 1
    prog.empty()

    n_eval = len(eval_rows)
    if n_eval == 0:
        st.info(f"No startable history to evaluate yet — pitchers need at least "
                f"{bt_min_prior} prior starts this season.")
    else:
        if opp_clean == 0:
            st.warning(
                f"📊 Opponent K% used current-season values for all {n_eval} starts — no "
                "pre-game team snapshots exist yet. The pitcher season line and recent form "
                "are still rebuilt cleanly from the game log; only the opponent-K% adjustment "
                "carries a small, stable look-ahead until daily snapshots accumulate."
            )
        else:
            pct = round(opp_clean / n_eval * 100)
            st.info(f"📸 {opp_clean}/{n_eval} starts ({pct}%) used a pre-game opponent-K% "
                    "snapshot; the rest fell back to current-season K%.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Starts evaluated", n_eval, f"{pitchers_evaluated} pitchers")
        c2.metric("Proj K — avg error", f"{round(sum(k_errs)/len(k_errs), 2)}")
        c3.metric("Proj ER — avg error", f"{round(sum(er_errs)/len(er_errs), 2)}")
        c4.metric("Proj IP — avg error", f"{round(sum(ip_errs)/len(ip_errs), 2)}")

        total_bets = sum(v["bets"] for v in line_stats.values())
        total_hits = sum(v["hits"] for v in line_stats.values())
        if total_bets:
            overall = round(total_hits / total_bets * 100, 1)
            st.markdown('<p class="section-head">Directional accuracy of O/U calls</p>',
                        unsafe_allow_html=True)
            st.caption(f"Across all reference lines: {total_hits}/{total_bets} calls correct "
                       f"({overall}%). Break-even at standard −110 juice is ~52.4%.")

            active = [ln for ln in COMMON_K_LINES if line_stats[ln]["bets"] > 0]
            if active:
                accs   = [round(line_stats[ln]["hits"]/line_stats[ln]["bets"]*100, 1) for ln in active]
                counts = [line_stats[ln]["bets"] for ln in active]
                figb = go.Figure()
                figb.add_trace(go.Bar(
                    x=[f"O/U {ln} ({c})" for ln, c in zip(active, counts)], y=accs,
                    marker_color=["#00c07a" if a >= 52.4 else "#ff5252" for a in accs],
                    text=[f"{a}%" for a in accs], textposition="outside",
                    textfont=dict(color="#ccc", size=11)))
                figb.add_hline(y=52.4, line_dash="dot", line_color="rgba(255,255,255,0.35)",
                               annotation_text="break-even ~52.4%",
                               annotation_font_color="rgba(255,255,255,0.4)",
                               annotation_position="right")
                figb.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=10, r=60, t=20, b=10), showlegend=False,
                    yaxis=dict(range=[0, 110], ticksuffix="%", tickfont=dict(color="#888", size=10),
                               showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    xaxis=dict(tickfont=dict(color="#ccc", size=11)))
                st.plotly_chart(figb, use_container_width=True)

            st.markdown('<p class="section-head">Directional accuracy by confidence tier</p>',
                        unsafe_allow_html=True)
            tcols  = st.columns(3)
            tcolor = {"High": "#00c07a", "Moderate": "#f5c842", "Low": "#f5a623"}
            for i, t in enumerate(["High", "Moderate", "Low"]):
                b = tier_stats[t]["bets"]; h = tier_stats[t]["hits"]
                acc = round(h / b * 100, 1) if b else 0
                with tcols[i]:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.04);border-left:4px solid {tcolor[t]};'
                        f'border-radius:8px;padding:12px 14px;">'
                        f'<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;'
                        f'color:{tcolor[t]};">{t}</div>'
                        f'<div style="font-size:22px;font-weight:800;color:'
                        f'{"#00c07a" if (acc>=52.4 and b) else "#ff5252"};">{acc if b else "—"}%</div>'
                        f'<div style="font-size:11px;color:#888;">{h}/{b} calls</div></div>',
                        unsafe_allow_html=True)
        else:
            st.caption("No actionable O/U calls were generated — every projection landed "
                       "within 1 K of the reference lines.")

        st.markdown('<p class="section-head">Recent evaluated starts</p>', unsafe_allow_html=True)
        eval_rows.sort(key=lambda r: r["date"], reverse=True)
        body = ""
        for r in eval_rows[:25]:
            diff = round(r["proj_k"] - r["actual_k"], 1)
            dcol = "#00c07a" if abs(diff) <= 1 else ("#f5c842" if abs(diff) <= 2 else "#ff5252")
            body += (
                f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
                f"<td style='padding:6px 10px;color:#888;font-size:12px;'>{r['date']}</td>"
                f"<td style='padding:6px 10px;color:#ccc;'>{r['pitcher']}</td>"
                f"<td style='padding:6px 10px;color:#888;font-size:12px;'>vs {r['opp']}</td>"
                f"<td style='padding:6px 10px;color:#f5c842;font-weight:700;'>{r['proj_k']}</td>"
                f"<td style='padding:6px 10px;color:#ccc;font-weight:700;'>{r['actual_k']}</td>"
                f"<td style='padding:6px 10px;color:{dcol};'>{'+' if diff>=0 else ''}{diff}</td>"
                f"<td style='padding:6px 10px;color:#888;font-size:11px;'>{r['tier']}</td></tr>")
        head = "".join(
            f"<th style='padding:6px 10px;color:#888;font-size:10px;letter-spacing:1.5px;"
            f"text-transform:uppercase;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1);'>{h}</th>"
            for h in ["Date", "Pitcher", "Opp", "Proj K", "Actual", "Δ", "Tier"])
        st.markdown(
            f"<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
            unsafe_allow_html=True)
        st.caption(f"Showing {min(25, n_eval)} of {n_eval} evaluated starts. "
                   "Metrics above reflect all of them.")

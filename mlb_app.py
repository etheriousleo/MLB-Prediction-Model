"""
MLB Win Probability — Today's Slate
-------------------------------------
Install dependencies:
    pip install MLB-StatsAPI streamlit plotly numpy

Run:
    streamlit run mlb_app.py
"""

import datetime
import json
import os
import numpy as np
import plotly.graph_objects as go
import statsapi
import streamlit as st

# ── Snapshot system ────────────────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def snapshot_path(date_str: str) -> str:
    """Return the file path for a given date's snapshot (YYYY-MM-DD)."""
    return os.path.join(SNAPSHOT_DIR, f"stats_{date_str}.json")

def save_snapshot(batting: dict, pitching: dict, standings: dict,
                  pitchers: dict = None, date_str: str = None,
                  batting_recent: dict = None, pitching_recent: dict = None,
                  recent_window: int = None):
    """
    Save today's team stats and SP stats to a JSON snapshot file.
    Called once per day automatically when the app loads.
    pitchers: {player_id: {era, whip, k9, ...}} for today's probable starters.
    batting_recent / pitching_recent: recent-form team stats as computed today.
    Storing these lets the backtest blend recent form WITHOUT look-ahead bias —
    a snapshot taken on day D holds recent form over [D-window, D], all of which
    predates any game played on D+1 or later.
    recent_window: the day-window (e.g. 14) used to compute the recent stats.
    """
    ensure_snapshot_dir()
    if date_str is None:
        date_str = datetime.datetime.today().strftime("%Y-%m-%d")
    path = snapshot_path(date_str)

    # If file exists but has no pitchers, update it — otherwise skip
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
            # Backfill recent form if this run has it and the file doesn't yet
            updated = False
            if batting_recent and not existing.get("batting_recent"):
                existing["batting_recent"]  = batting_recent
                existing["pitching_recent"] = pitching_recent or {}
                existing["recent_window"]   = recent_window
                updated = True
            if existing.get("pitchers") or not pitchers:
                if updated:
                    with open(path, "w") as f:
                        json.dump(existing, f)
                return  # Already complete (or nothing new to add)
            # Has team stats but no pitchers yet — add them
            existing["pitchers"] = {str(k): v for k, v in (pitchers or {}).items()}
            with open(path, "w") as f:
                json.dump(existing, f)
            return
        except Exception:
            pass

    payload = {
        "date":     date_str,
        "batting":  batting,
        "pitching": pitching,
        "standings": standings,
        "pitchers": {str(k): v for k, v in (pitchers or {}).items()},
        "batting_recent":  batting_recent or {},
        "pitching_recent": pitching_recent or {},
        "recent_window":   recent_window,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass  # Fail silently — snapshot is best-effort


def load_sp_snapshot(date_str: str) -> dict:
    """
    Load the pitcher stats from a snapshot for a given date.
    Returns {player_id_str: stats_dict} or {} if not found.
    """
    path = snapshot_path(date_str)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("pitchers", {})
    except Exception:
        return {}

def load_recent_snapshot(date_str: str) -> tuple[dict, dict]:
    """
    Load snapshotted recent-form team stats for a given date.
    Returns (batting_recent, pitching_recent) or ({}, {}) if not present.
    Older snapshots saved before recent-form snapshotting existed simply
    return ({}, {}), in which case the backtest falls back to season-only
    for that game — no look-ahead, just no recent signal.
    """
    path = snapshot_path(date_str)
    if not os.path.exists(path):
        return {}, {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("batting_recent", {}), payload.get("pitching_recent", {})
    except Exception:
        return {}, {}

def load_snapshot(date_str: str) -> tuple[dict, dict, dict]:
    """
    Load the snapshot for a given date.
    Returns (batting, pitching, standings) or (None, None, None) if not found.
    """
    path = snapshot_path(date_str)
    if not os.path.exists(path):
        return None, None, None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload["batting"], payload["pitching"], payload.get("standings", {})
    except Exception:
        return None, None, None


def get_sp_stats_from_cache(snap_cache_pitchers: dict, player_id) -> dict:
    """
    Look up a pitcher's stats from the snapshot pitcher cache.
    Handles both int and string keys since JSON keys are always strings.
    """
    if not player_id or not snap_cache_pitchers:
        return {}
    return (snap_cache_pitchers.get(str(player_id)) or
            snap_cache_pitchers.get(int(player_id) if str(player_id).isdigit() else player_id) or {})

def get_best_snapshot_for_game(game_date_str: str) -> tuple[dict, dict, dict, str]:
    """
    Find the most recent snapshot taken BEFORE a game's date.
    This ensures we use stats as they existed before the game was played.
    Returns (batting, pitching, standings, snapshot_date) or (None,None,None,None).
    """
    ensure_snapshot_dir()
    try:
        game_dt = datetime.datetime.strptime(game_date_str, "%Y-%m-%d")
    except ValueError:
        return None, None, None, None

    # List all snapshots, find most recent one before game date
    snapshot_files = sorted([
        f for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("stats_") and f.endswith(".json")
    ], reverse=True)

    for fname in snapshot_files:
        snap_date_str = fname.replace("stats_", "").replace(".json", "")
        try:
            snap_dt = datetime.datetime.strptime(snap_date_str, "%Y-%m-%d")
            if snap_dt < game_dt:
                b, p, s = load_snapshot(snap_date_str)
                if b and p:
                    return b, p, s, snap_date_str
        except ValueError:
            continue

    return None, None, None, None

def list_snapshots() -> list[str]:
    """Return sorted list of available snapshot dates."""
    ensure_snapshot_dir()
    files = [
        f.replace("stats_", "").replace(".json", "")
        for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("stats_") and f.endswith(".json")
    ]
    return sorted(files)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MLB Daily Slate", page_icon="⚾", layout="wide")

st.markdown("""
<style>
    .section-head {
        font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
        color: #888; margin: 1.5rem 0 0.5rem;
    }
    .stat-better { color: #00c07a; font-weight: 600; }
    .stat-worse  { color: #ff5252; }
    .confidence-reason { font-size: 0.85rem; line-height: 1.6; color: #ccc; }
</style>
""", unsafe_allow_html=True)

SEASON     = datetime.datetime.now().year
HOME_BOOST = 0.04
HOME_RUNS  = 0.3
# Parks more than 5% above/below neutral already encode much of the home
# environment in their run factor.  Scale the win-probability home boost
# toward neutral for those venues to avoid double-counting.
def scaled_home_boost(park_factor: float = 1.0) -> float:
    """Return a park-adjusted HOME_BOOST (win-prob units)."""
    # Linear taper: full boost at pf=1.0, half boost at pf ≥ 1.10 or ≤ 0.90
    deviation = abs(park_factor - 1.0)
    scale = max(0.5, 1.0 - deviation * 5)   # e.g. pf=1.10 → scale=0.5
    return HOME_BOOST * scale

# ── Team lookup tables ─────────────────────────────────────────────────────────
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
    "Arizona Diamondbacks":  "Arizona Diamondbacks",
    "Atlanta Braves":        "Atlanta Braves",
    "Baltimore Orioles":     "Baltimore Orioles",
    "Boston Red Sox":        "Boston Red Sox",
    "Chicago Cubs":          "Chicago Cubs",
    "Chicago White Sox":     "Chicago White Sox",
    "Cincinnati Reds":       "Cincinnati Reds",
    "Cleveland Guardians":   "Cleveland Guardians",
    "Colorado Rockies":      "Colorado Rockies",
    "Detroit Tigers":        "Detroit Tigers",
    "Houston Astros":        "Houston Astros",
    "Kansas City Royals":    "Kansas City Royals",
    "Los Angeles Angels":    "Los Angeles Angels",
    "Los Angeles Dodgers":   "Los Angeles Dodgers",
    "Miami Marlins":         "Miami Marlins",
    "Milwaukee Brewers":     "Milwaukee Brewers",
    "Minnesota Twins":       "Minnesota Twins",
    "New York Mets":         "New York Mets",
    "New York Yankees":      "New York Yankees",
    "Oakland Athletics":     "Oakland Athletics",
    "Athletics":             "Oakland Athletics",
    "Sacramento Athletics":  "Oakland Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates":    "Pittsburgh Pirates",
    "San Diego Padres":      "San Diego Padres",
    "San Francisco Giants":  "San Francisco Giants",
    "Seattle Mariners":      "Seattle Mariners",
    "St. Louis Cardinals":   "St. Louis Cardinals",
    "Tampa Bay Rays":        "Tampa Bay Rays",
    "Texas Rangers":         "Texas Rangers",
    "Toronto Blue Jays":     "Toronto Blue Jays",
    "Washington Nationals":  "Washington Nationals",
}


# ── Ballpark factors ──────────────────────────────────────────────────────────
# Run factor: multiplier applied to projected runs at this venue.
# Based on multi-year park factor data (1.00 = league average).
# Sources: FanGraphs park factors, Baseball Reference.
# Coors is the most extreme; sea-level domes are the most suppressive.
PARK_RUN_FACTOR = {
    # Hitter-friendly
    "COL": 1.18,   # Coors Field — altitude massively inflates offense
    "CIN": 1.07,   # Great American Ball Park — small dimensions
    "TEX": 1.06,   # Globe Life Field — hot/dry air, hitter dimensions
    "BOS": 1.05,   # Fenway Park — Green Monster, short LF
    "CHC": 1.04,   # Wrigley Field — wind-dependent but avg hitter-friendly
    "BAL": 1.04,   # Camden Yards — intimate dimensions
    "PHI": 1.03,   # Citizens Bank Park
    "MIL": 1.03,   # American Family Field — dome effect on warmer days
    "ARI": 1.03,   # Chase Field — hot dry air, retractable roof
    "NYY": 1.02,   # Yankee Stadium — short RF porch
    "ATL": 1.02,   # Truist Park
    "MIN": 1.01,   # Target Field
    "DET": 1.01,   # Comerica Park
    "HOU": 1.01,   # Minute Maid Park
    # Neutral
    "LAD": 1.00,   # Dodger Stadium
    "SFG": 1.00,   # Oracle Park — cold/wind cancels out
    "WSN": 1.00,   # Nationals Park
    "STL": 1.00,   # Busch Stadium
    "PIT": 0.99,   # PNC Park
    "TOR": 0.99,   # Rogers Centre — dome
    "NYM": 0.99,   # Citi Field — spacious
    "KCR": 0.99,   # Kauffman Stadium — large outfield
    "CLE": 0.99,   # Progressive Field
    "CHW": 0.98,   # Guaranteed Rate Field
    # Pitcher-friendly
    "SEA": 0.97,   # T-Mobile Park — marine air, large dimensions
    "LAA": 0.97,   # Angel Stadium
    "OAK": 0.97,   # Oakland Coliseum — large foul territory, marine air
    "SDP": 0.96,   # Petco Park — spacious, marine layer suppresses HR
    "MIA": 0.96,   # LoanDepot Park — air-conditioned dome, deep dimensions
    "TBR": 0.95,   # Tropicana Field — dome, largest dimensions in AL
}

# Venue ID → team abbreviation (home team's park)
# These are stable statsapi venue IDs
VENUE_ID_TO_ABB = {
    # Confirmed against MLB Stats API / Baseball Savant venueId values.
    # ballpark_factor() falls back to home_abb if a venue_id isn't listed here.
    1:    "LAA",  # Angel Stadium
    2:    "BAL",  # Oriole Park at Camden Yards
    3:    "BOS",  # Fenway Park
    4:    "CHW",  # Guaranteed Rate Field
    5:    "CLE",  # Progressive Field
    7:    "KCR",  # Kauffman Stadium
    10:   "OAK",  # Oakland Coliseum
    12:   "TBR",  # Tropicana Field
    14:   "TOR",  # Rogers Centre
    15:   "ARI",  # Chase Field
    17:   "CHC",  # Wrigley Field
    31:   "PIT",  # PNC Park
    32:   "MIL",  # American Family Field
    239:  "LAD",  # Dodger Stadium
    680:  "SEA",  # T-Mobile Park
    2392: "COL",  # Coors Field
    2394: "DET",  # Comerica Park (confirmed: Baseball Savant venueId=2394)
    2395: "SFG",  # Oracle Park
    2430: "CIN",  # Great American Ball Park
    2680: "SDP",  # Petco Park
    2681: "PHI",  # Citizens Bank Park
    2889: "STL",  # Busch Stadium
    3289: "NYM",  # Citi Field
    3309: "WSN",  # Nationals Park
    3312: "MIN",  # Target Field
    3313: "NYY",  # Yankee Stadium
    4169: "MIA",  # loanDepot park
    4705: "ATL",  # Truist Park
    4707: "HOU",  # Minute Maid Park / Daikin Park
    5325: "TEX",  # Globe Life Field
}

def ballpark_factor(venue_id: int, home_abb: str) -> float:
    """
    Return the run-scoring multiplier for this venue.
    Falls back to home team's park if venue_id not in map,
    then to 1.0 (neutral) if neither matches.
    """
    if venue_id and venue_id in VENUE_ID_TO_ABB:
        abb = VENUE_ID_TO_ABB[venue_id]
        return PARK_RUN_FACTOR.get(abb, 1.00)
    return PARK_RUN_FACTOR.get(home_abb, 1.00)


# ── Data fetching ──────────────────────────────────────────────────────────────
def api_name_to_abb(name: str):
    return FULL_TO_ABB.get(STATSAPI_NAME_MAP.get(name, name))


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_stats(season: int) -> tuple[dict, dict]:
    all_batting, all_pitching = {}, {}
    try:
        teams_resp = statsapi.get("teams", {"sportId": 1, "season": season})
        team_list  = [t for t in teams_resp.get("teams", [])
                      if t.get("sport", {}).get("id") == 1]
    except Exception:
        return all_batting, all_pitching

    for team in team_list:
        tid  = team["id"]
        abb  = FULL_TO_ABB.get(STATSAPI_NAME_MAP.get(team.get("name", ""), team.get("name", "")))
        if not abb:
            continue
        for group, store in [("hitting", all_batting), ("pitching", all_pitching)]:
            try:
                raw    = statsapi.get("team_stats", {"teamId": tid, "stats": "season",
                                                     "group": group, "season": season})
                splits = raw.get("stats", [{}])[0].get("splits", [])
                s      = splits[0].get("stat", {}) if splits else {}
                if not s:
                    continue
                if group == "hitting":
                    g   = max(int(s.get("gamesPlayed", 0) or 0), 1)
                    pa  = max(float(s.get("plateAppearances", 0) or 0), 1)
                    bb  = float(s.get("baseOnBalls",   0) or 0)
                    so  = float(s.get("strikeOuts",    0) or 0)
                    store[abb] = {
                        "G":      g,
                        "R":      float(s.get("runs",     0) or 0),
                        "BA":     float(s.get("avg",      0) or 0),
                        "OBP":    float(s.get("obp",      0) or 0),
                        "SLG":    float(s.get("slg",      0) or 0),
                        "OPS":    float(s.get("ops",      0) or 0),
                        "HR":     float(s.get("homeRuns", 0) or 0),
                        "BB_pct": round(bb / pa, 3),       # walk rate
                        "K_pct":  round(so / pa, 3),       # strikeout rate (lower = better for hitters)
                        "RD":     0.0,  # populated from standings in score_team
                    }
                else:
                    bb9  = float(s.get("walksPer9Inn",       "0") or 0)
                    k9   = float(s.get("strikeoutsPer9Inn",  "0") or 0)
                    hr9  = float(s.get("homeRunsPer9",       "0") or 0)
                    era  = float(s.get("era",  0) or 0)
                    whip = float(s.get("whip", 0) or 0)
                    # FIP per-9 approximation: numerically equivalent to standard FIP
                    # when stats are expressed per 9 IP. Constant 3.10 matches league
                    # average ERA scaling (standard range ~3.1–3.2). Clamped to
                    # [2.0, 7.5] to prevent extreme early-season values from
                    # dominating the defence score (FIP weight = 35%).
                    fip  = round(max(2.0, min(7.5,
                               (13 * hr9 + 3 * bb9 - 2 * k9) / 9 + 3.10)), 2)
                    store[abb] = {
                        "ERA":  era,
                        "WHIP": whip,
                        "K/9":  k9,
                        "BB/9": bb9,
                        "HR/9": hr9,
                        "FIP":  fip,
                    }
            except Exception:
                pass
    return all_batting, all_pitching


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_team_stats_recent(season: int, last_n: int) -> tuple[dict, dict]:
    """
    Compute recent team stats from actual boxscores of completed games.
    Uses statsapi.schedule + per-game boxscores — reliable, no byDateRange needed.
    """
    from collections import defaultdict
    all_batting  = {}
    all_pitching = {}

    end_dt    = datetime.datetime.today()
    start_dt  = max(end_dt - datetime.timedelta(days=last_n),
                    datetime.datetime(season, 3, 20))
    start_str = start_dt.strftime("%m/%d/%Y")
    end_str   = end_dt.strftime("%m/%d/%Y")

    try:
        games     = statsapi.schedule(start_date=start_str, end_date=end_str, sportId=1)
        completed = [g for g in games
                     if g.get("status") == "Final" and g.get("game_type") == "R"]
    except Exception:
        return all_batting, all_pitching

    if not completed:
        return all_batting, all_pitching

    team_h = defaultdict(lambda: {"G":0,"R":0,"H":0,"AB":0,"BB":0,"SO":0,"HR":0,"2B":0,"3B":0})
    team_p = defaultdict(lambda: {"G":0,"ER":0,"IP":0.0,"H":0,"BB":0,"SO":0,"HR":0})

    for game in completed:
        gid = game.get("game_id")
        if not gid:
            continue
        try:
            box       = statsapi.get("game", {"gamePk": gid})
            teams_box = box.get("liveData",{}).get("boxscore",{}).get("teams",{})
            for side in ["home","away"]:
                tb    = teams_box.get(side, {})
                tname = tb.get("team",{}).get("name","")
                abb   = FULL_TO_ABB.get(STATSAPI_NAME_MAP.get(tname, tname))
                if not abb:
                    continue
                bs = tb.get("teamStats",{}).get("batting",{})
                team_h[abb]["G"]  += 1
                team_h[abb]["R"]  += int(bs.get("runs",        0) or 0)
                team_h[abb]["H"]  += int(bs.get("hits",        0) or 0)
                team_h[abb]["AB"] += int(bs.get("atBats",      0) or 0)
                team_h[abb]["BB"] += int(bs.get("baseOnBalls", 0) or 0)
                team_h[abb]["SO"] += int(bs.get("strikeOuts",  0) or 0)
                team_h[abb]["HR"] += int(bs.get("homeRuns",    0) or 0)
                team_h[abb]["2B"] += int(bs.get("doubles",     0) or 0)
                team_h[abb]["3B"] += int(bs.get("triples",     0) or 0)
                ps = tb.get("teamStats",{}).get("pitching",{})
                try:
                    ip = float(str(ps.get("inningsPitched","0") or "0"))
                except ValueError:
                    ip = 0.0
                team_p[abb]["G"]  += 1
                team_p[abb]["ER"] += int(ps.get("earnedRuns",  0) or 0)
                team_p[abb]["IP"] += ip
                team_p[abb]["H"]  += int(ps.get("hits",        0) or 0)
                team_p[abb]["BB"] += int(ps.get("baseOnBalls", 0) or 0)
                team_p[abb]["SO"] += int(ps.get("strikeOuts",  0) or 0)
                team_p[abb]["HR"] += int(ps.get("homeRuns",    0) or 0)
        except Exception:
            continue

    for abb, th in team_h.items():
        g   = max(th["G"], 1)
        ab  = max(th["AB"], 1)
        pa  = max(ab + th["BB"], 1)
        ba  = round(th["H"] / ab, 3)
        obp = round((th["H"] + th["BB"]) / pa, 3)
        # Real SLG from total bases. Singles = hits minus extra-base hits.
        # If the boxscore omits doubles/triples, they read as 0, so total bases
        # degrade gracefully to H + 3*HR (all non-HR hits treated as singles) —
        # still grounded in actual outcomes, unlike a flat BA multiplier.
        singles = max(th["H"] - th["2B"] - th["3B"] - th["HR"], 0)
        tb      = singles + 2 * th["2B"] + 3 * th["3B"] + 4 * th["HR"]
        slg = round(tb / ab, 3)
        ops = round(obp + slg, 3)
        all_batting[abb] = {
            "G":      g,
            "R":      th["R"],
            "BA":     ba,
            "OBP":    obp,
            "SLG":    slg,
            "OPS":    ops,
            "HR":     th["HR"],
            "BB_pct": round(th["BB"] / pa, 3),
            "K_pct":  round(th["SO"] / pa, 3),
            "RD":     0.0,
        }

    for abb, tp in team_p.items():
        ip   = max(tp["IP"], 0.1)
        bb9  = round((tp["BB"] * 9) / ip, 2)
        k9   = round((tp["SO"] * 9) / ip, 1)
        hr9  = round((tp["HR"] * 9) / ip, 2)
        all_pitching[abb] = {
            "ERA":  round((tp["ER"] * 9) / ip, 2),
            "WHIP": round((tp["H"] + tp["BB"]) / ip, 3),
            "K/9":  k9,
            "BB/9": bb9,
            "HR/9": hr9,
            "FIP":  round(max(2.0, min(7.5, (13 * hr9 + 3 * bb9 - 2 * k9) / 9 + 3.10)), 2),
        }

    return all_batting, all_pitching

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_standings() -> dict:
    records = {}
    try:
        for _, div_info in statsapi.standings_data().items():
            for team in div_info.get("teams", []):
                name = STATSAPI_NAME_MAP.get(team.get("name", ""), team.get("name", ""))
                w    = int(team.get("w", 0))
                l    = int(team.get("l", 0))
                gp   = w + l
                records[name] = {
                    "W":     w,
                    "L":     l,
                    "W_PCT": w / gp if gp > 0 else 0.5,
                    "RD":    0,     # enriched later by enrich_standings_with_rd()
                    "RD_PG": 0.0,
                }
    except Exception:
        pass
    return records


def enrich_standings_with_rd(standings: dict, batting: dict) -> dict:
    """
    Compute run differential per game using runs scored (from batting stats)
    and runs allowed (from pitching stats), then inject into standings dict.
    Called after both batting and pitching data are available.
    """
    # Build a runs-scored lookup: full_name -> runs_per_game
    for abb, b in batting.items():
        full_name = ABB_TO_FULL.get(abb, abb)
        if full_name not in standings:
            continue
        gp = max(float(b.get("G", 1) or 1), 1)
        rs = float(b.get("R", 0) or 0)   # runs scored
        # We don't have runs allowed directly, but we can approximate:
        # runs_allowed ≈ ERA * IP / 9, but simpler: use W/L record to estimate
        # Best available: store runs scored per game and derive RD from schedule
        # For now store RS/G — we'll compute RD via schedule separately
        standings[full_name]["RS_PG"] = round(rs / gp, 2)
    return standings


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_run_differential(season: int) -> dict:
    """
    Compute each team's run differential by summing all completed game scores.
    Returns {full_name: {"rd": int, "rd_pg": float, "gp": int}}
    """
    result = {}
    from collections import defaultdict
    team_runs = defaultdict(lambda: {"rs": 0, "ra": 0, "gp": 0})
    try:
        start = datetime.datetime(season, 3, 20).strftime("%m/%d/%Y")
        end   = datetime.datetime.today().strftime("%m/%d/%Y")
        games = statsapi.schedule(start_date=start, end_date=end, sportId=1)
        completed = [g for g in games
                     if g.get("status") == "Final" and g.get("game_type") == "R"]
        for g in completed:
            hs = int(g.get("home_score", 0) or 0)
            as_ = int(g.get("away_score", 0) or 0)
            hn  = STATSAPI_NAME_MAP.get(g.get("home_name",""), g.get("home_name",""))
            an  = STATSAPI_NAME_MAP.get(g.get("away_name",""), g.get("away_name",""))
            team_runs[hn]["rs"] += hs
            team_runs[hn]["ra"] += as_
            team_runs[hn]["gp"] += 1
            team_runs[an]["rs"] += as_
            team_runs[an]["ra"] += hs
            team_runs[an]["gp"] += 1
    except Exception:
        pass
    for name, r in team_runs.items():
        gp = max(r["gp"], 1)
        rd = r["rs"] - r["ra"]
        result[name] = {
            "rd":    rd,
            "rd_pg": round(rd / gp, 3),
            "gp":    r["gp"],
        }
    return result


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_todays_games() -> tuple[list, str]:
    today = datetime.datetime.today().strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(date=today, sportId=1)
        return [g for g in games if g.get("game_type", "") == "R"], ""
    except Exception as e:
        return [], str(e)


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_season_games(season: int) -> tuple[list, str]:
    start = datetime.datetime(season, 3, 20).strftime("%m/%d/%Y")
    end   = datetime.datetime.today().strftime("%m/%d/%Y")
    try:
        games     = statsapi.schedule(start_date=start, end_date=end, sportId=1)
        completed = [g for g in games
                     if g.get("status", "") == "Final"
                     and g.get("game_type", "") == "R"]
        return completed, ""
    except Exception as e:
        return [], str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_season_stats(player_id: int, season: int) -> dict:
    """Fetch a pitcher's season stats by player ID via the MLB Stats API."""
    try:
        # Correct endpoint: people/{id}/stats
        raw = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=pitching,type=season,season={season})",
        })
        people = raw.get("people", [])
        if not people:
            return {}
        stats_list = people[0].get("stats", [])
        if not stats_list:
            return {}
        splits = stats_list[0].get("splits", [])
        s = splits[0].get("stat", {}) if splits else {}
        if not s:
            return {}
        bb9 = float(s.get("walksPer9Inn", "0") or 0)
        k9  = float(s.get("strikeoutsPer9Inn", "0") or 0)
        era = float(s.get("era",  "0") or 0)
        whip= float(s.get("whip", "0") or 0)
        ip  = float(s.get("inningsPitched", "0") or 0)
        gs  = int(s.get("gamesStarted", 0))
        # FIP approximation for individual pitcher
        hr9 = float(s.get("homeRunsPer9", "0") or 0)
        fip = round(max(2.0, min(7.5,
                    (13 * hr9 + 3 * bb9 - 2 * k9) / 9 + 3.10)), 2) if k9 else era
        return {
            "era":    era,
            "whip":   whip,
            "k9":     k9,
            "bb9":    bb9,
            "fip":    fip,
            "ip":     ip,
            "gs":     gs,
            "wins":   int(s.get("wins",   0)),
            "losses": int(s.get("losses", 0)),
        }
    except Exception:
        return {}


# ── Model helpers ──────────────────────────────────────────────────────────────
def sf(d: dict, *keys) -> float:
    for k in keys:
        try:
            v = float(d.get(k, 0) or 0)
            if v: return v
        except (TypeError, ValueError):
            pass
    return 0.0


# League-average baselines for absolute normalisation
LEAGUE_AVG = {
    "ops": 0.720, "obp": 0.317, "slg": 0.403,
    "runs_pg": 4.5, "bb_pct": 0.085, "k_pct": 0.225,
    "era": 4.20,  "whip": 1.28,  "k9": 8.7,
    "fip": 4.10,  "bb9": 3.1,
    "rd_pg": 0.0,   # run differential per game vs league avg
}

def norm_vs_league(val: float, avg: float, std: float, inv: bool = False) -> float:
    """Normalise a stat against a league-average baseline. Returns 0-1, 0.5=league avg."""
    z = (val - avg) / std if std else 0
    z = max(-3, min(3, z))
    score = (z + 3) / 6          # map -3..+3 → 0..1
    return round(1 - score if inv else score, 3)

LEAGUE_STD = {
    "ops": 0.065, "obp": 0.025, "slg": 0.050,
    "runs_pg": 0.7, "bb_pct": 0.020, "k_pct": 0.035,
    "era": 0.80,  "whip": 0.15,  "k9": 1.5,
    "fip": 0.75,  "bb9": 0.7,
    "rd_pg": 0.5,
}

def score_team(abb: str, batting: dict, pitching: dict, standings: dict) -> dict:
    b         = batting.get(abb, {})
    p         = pitching.get(abb, {})
    games_b   = max(float(b.get("G", 1) or 1), 1)
    full_name = ABB_TO_FULL.get(abb, abb)
    rec       = standings.get(full_name, {})
    w         = rec.get("W", 0)
    l         = rec.get("L", 0)
    wpct      = rec.get("W_PCT", 0.5)
    # RD comes from standings — the hitting API does not return this field
    rd_pg     = rec.get("RD_PG", 0.0)
    return {
        "name":    full_name,
        "abb":     abb,
        "G":       games_b,
        "runs_pg": round(sf(b, "R") / games_b, 2),
        "avg":     round(sf(b, "BA",  "AVG"), 3),
        "obp":     round(sf(b, "OBP"), 3),
        "slg":     round(sf(b, "SLG"), 3),
        "ops":     round(sf(b, "OPS"), 3),
        "hr_pg":   round(sf(b, "HR") / games_b, 2),
        "bb_pct":  round(sf(b, "BB_pct"), 3),
        "k_pct":   round(sf(b, "K_pct"),  3),   # lower is better for hitters
        "rd_pg":   rd_pg,
        "era":     round(sf(p, "ERA"), 2),
        "whip":    round(sf(p, "WHIP"), 3),
        "k9":      round(sf(p, "K/9", "SO9"), 1),
        "bb9":     round(sf(p, "BB/9"), 2),
        "fip":     round(sf(p, "FIP") or sf(p, "ERA"), 2),  # fallback to ERA
        "w": w, "l": l, "wpct": round(wpct * 100, 1),
    }


def blend_stats(season_s: dict, recent_s: dict, w_season: float, w_recent: float) -> dict:
    """
    Blend season and recent stats across all key metrics.
    Uses G (games played) as the reliability gate — not runs_pg which can be 0.
    Only blends a stat if the recent value is non-zero to avoid dragging good
    season numbers down toward zero when the API returns incomplete data.
    """
    if not recent_s:
        return season_s
    recent_games = float(recent_s.get("G", 0) or recent_s.get("g", 0) or 0)
    if recent_games < 1:
        return season_s
    blended = dict(season_s)
    for key in ("runs_pg", "avg", "obp", "slg", "ops", "hr_pg",
                "bb_pct", "k_pct", "rd_pg",
                "era", "whip", "k9", "bb9", "fip"):
        sv = season_s.get(key, 0) or 0
        rv = recent_s.get(key, 0) or 0
        if rv != 0:
            blended[key] = round(sv * w_season + rv * w_recent, 3)
    return blended


def score_vs_league(s: dict) -> dict:
    """
    Convert a team's raw stats into 0-1 scores vs league average.
    0.5 = exactly league average. 1.0 = best possible. 0.0 = worst possible.
    This prevents two bad teams from both looking "average" relative to each other.
    """
    def n(key, inv=False):
        return norm_vs_league(s.get(key, LEAGUE_AVG[key]),
                              LEAGUE_AVG[key], LEAGUE_STD[key], inv)
    return {
        # Offense (higher is better)
        "ops_score":    n("ops"),
        "obp_score":    n("obp"),
        "slg_score":    n("slg"),
        "runs_score":   n("runs_pg"),
        "bb_score":     n("bb_pct"),          # drawing walks = good
        "k_hit_score":  n("k_pct",  inv=True),# striking out less = good for hitters
        "rd_score":     n("rd_pg"),           # run differential per game
        # Pitching / defence (lower ERA/WHIP/FIP/BB9 is better)
        "era_score":    n("era",  inv=True),
        "whip_score":   n("whip", inv=True),
        "k9_score":     n("k9"),
        "fip_score":    n("fip",  inv=True),  # FIP more predictive than ERA
        "bb9_score":    n("bb9",  inv=True),  # fewer walks allowed = good
        # Record
        "wpct_score":   min(1.0, max(0.0, s.get("wpct", 50) / 100)),
        "rd_rec_score": n("rd_pg"),           # used separately in record component
    }


def build_composite(sa: dict, sb: dict):
    """
    Score each team vs league average, then return (off_a, def_a, rec_a, off_b, def_b, rec_b).
    Using absolute league-average normalisation means both teams can score below 0.5
    if they're both bad, giving a more accurate win probability.
    """
    la = score_vs_league(sa)
    lb = score_vs_league(sb)

    # Offense: OPS (40%), OBP (20%), runs/g (20%), BB% (10%), K% hitter (10%)
    off_a = (la["ops_score"]   * 0.40 + la["obp_score"]   * 0.20 +
             la["runs_score"]  * 0.20 + la["bb_score"]    * 0.10 +
             la["k_hit_score"] * 0.10)
    off_b = (lb["ops_score"]   * 0.40 + lb["obp_score"]   * 0.20 +
             lb["runs_score"]  * 0.20 + lb["bb_score"]    * 0.10 +
             lb["k_hit_score"] * 0.10)

    # Defence/pitching: FIP (35%), ERA (25%), WHIP (20%), K/9 (10%), BB/9 (10%)
    def_a = (la["fip_score"]  * 0.35 + la["era_score"]  * 0.25 +
             la["whip_score"] * 0.20 + la["k9_score"]   * 0.10 +
             la["bb9_score"]  * 0.10)
    def_b = (lb["fip_score"]  * 0.35 + lb["era_score"]  * 0.25 +
             lb["whip_score"] * 0.20 + lb["k9_score"]   * 0.10 +
             lb["bb9_score"]  * 0.10)

    # Record: W% (60%) + run differential per game (40%) — RD is more predictive than W%
    rec_a = la["wpct_score"] * 0.60 + la["rd_score"] * 0.40
    rec_b = lb["wpct_score"] * 0.60 + lb["rd_score"] * 0.40

    return off_a, def_a, rec_a, off_b, def_b, rec_b


def pitcher_adjustment(pitcher_stats: dict) -> float:
    """
    Return a pitching quality score 0–1 based on ERA, WHIP, K/9, BB/9.
    0.5 = league average. Higher = better pitcher (suppresses opponent more).
    Uses league-average normalisation so scores are meaningful in absolute terms.
    Only applied if pitcher has meaningful innings pitched.
    """
    if not pitcher_stats or pitcher_stats.get("ip", 0) < 15:
        return 0.5
    era  = pitcher_stats.get("era",  LEAGUE_AVG["era"])
    whip = pitcher_stats.get("whip", LEAGUE_AVG["whip"])
    k9   = pitcher_stats.get("k9",   LEAGUE_AVG["k9"])
    # FIP from individual pitcher stats: need K, BB, HR per 9
    # Approximate FIP from ERA + K9 signal
    bb   = pitcher_stats.get("bb9",  LEAGUE_AVG["bb9"])

    era_score  = norm_vs_league(era,  LEAGUE_AVG["era"],  LEAGUE_STD["era"],  inv=True)
    whip_score = norm_vs_league(whip, LEAGUE_AVG["whip"], LEAGUE_STD["whip"], inv=True)
    k9_score   = norm_vs_league(k9,   LEAGUE_AVG["k9"],   LEAGUE_STD["k9"])
    bb9_score  = norm_vs_league(bb,   LEAGUE_AVG["bb9"],  LEAGUE_STD["bb9"],  inv=True)

    # ERA (30%) + WHIP (30%) + K/9 (25%) + BB/9 (15%)
    return round(era_score * 0.30 + whip_score * 0.30 +
                 k9_score  * 0.25 + bb9_score  * 0.15, 3)


def calc_prob(sa, sb, home, w_off, w_def, w_rec,
              ra=None, rb=None, w_season=1.0, w_recent=0.0,
              sp_h_score=0.5, sp_a_score=0.5, park_factor: float = 1.0):
    ea = blend_stats(sa, ra, w_season, w_recent) if ra else sa
    eb = blend_stats(sb, rb, w_season, w_recent) if rb else sb

    off_a, def_a, rec_a, off_b, def_b, rec_b = build_composite(ea, eb)
    sc_a = off_a * w_off + def_a * w_def + rec_a * w_rec
    sc_b = off_b * w_off + def_b * w_def + rec_b * w_rec

    # SP adjustment: apply directly as an additive shift on the composite score.
    # sp_score is 0–1 (0.5 = league average). We convert the deviation from 0.5
    # into a bonus/penalty capped at ±0.04 (same order as HOME_BOOST).
    # Home SP boosts home team; away SP boosts away team.
    # This avoids the previous circular approach of re-deriving fake ERA/WHIP/K9
    # values from the SP score and feeding them back through normalization.
    SP_MAX = 0.04
    sp_h_adj = (sp_h_score - 0.5) * 2 * SP_MAX   # positive = home SP is above avg
    sp_a_adj = (sp_a_score - 0.5) * 2 * SP_MAX   # positive = away SP is above avg
    sc_a += sp_h_adj   # better home SP helps home team
    sc_b += sp_a_adj   # better away SP helps away team

    boost = scaled_home_boost(park_factor)
    if home == "home":   sc_a += boost
    elif home == "away": sc_b += boost
    total = sc_a + sc_b or 1
    return sc_a / total, sc_b / total


def calc_run_line(sa, sb, home, w_off, w_def, w_rec,
                  ra=None, rb=None, w_season=1.0, w_recent=0.0,
                  sp_h_score=0.5, sp_a_score=0.5,
                  park_factor: float = 1.0):
    ea = blend_stats(sa, ra, w_season, w_recent) if ra else sa
    eb = blend_stats(sb, rb, w_season, w_recent) if rb else sb
    _, def_a, _, _, def_b, _ = build_composite(ea, eb)

    # SP adjustment: elite SP suppresses opponent runs by up to 1.5 runs
    sp_h_supp = (sp_h_score - 0.5) * 3.0  # home SP suppresses away offense
    sp_a_supp = (sp_a_score - 0.5) * 3.0  # away SP suppresses home offense

    supp   = 1.5
    proj_h = ea["runs_pg"] - (def_b - 0.5) * supp * 2 - sp_a_supp
    proj_a = eb["runs_pg"] - (def_a - 0.5) * supp * 2 - sp_h_supp

    if home == "home": proj_h += HOME_RUNS

    # Apply ballpark factor — scales both teams' projected runs equally
    proj_h = proj_h * park_factor
    proj_a = proj_a * park_factor

    proj_h = max(1.5, proj_h)
    proj_a = max(1.5, proj_a)
    raw_margin = proj_h - proj_a
    winner = sa["name"] if raw_margin >= 0 else sb["name"]
    # Snap individual scores and total to nearest 0.5 — cleaner and more readable
    def snap(val): return round(round(val * 2) / 2, 1)
    proj_h     = snap(proj_h)
    proj_a     = snap(proj_a)
    proj_total = snap(proj_h + proj_a)
    # Projected margin = how many runs the favorite is projected to win by,
    # derived from the SNAPPED scores so it's consistent with the projected
    # score shown on the card. (Previously this slot returned the constant 1.5 —
    # the sportsbook run line — so every card displayed "1.5" and the backtest's
    # run-line error measured distance from 1.5 instead of from the projection.)
    proj_margin = round(abs(proj_h - proj_a), 1)
    return proj_h, proj_a, proj_margin, winner, raw_margin, proj_total


def calc_confidence(sa, sb, pct_h, pct_a, margin_winner, prob_winner,
                    ra=None, rb=None, sp_h_score=0.5, sp_a_score=0.5):
    models_agree  = prob_winner == margin_winner
    prob_gap      = abs(pct_h - pct_a)
    prob_strength = "strong" if prob_gap >= 12 else ("moderate" if prob_gap >= 5 else "narrow")

    ops_leader    = sa["name"] if sa["ops"]  > sb["ops"]  else sb["name"]
    rec_leader    = sa["name"] if sa["wpct"] > sb["wpct"] else sb["name"]
    ops_rec_split = ops_leader != rec_leader

    form_split = False
    if ra and rb:
        season_leader = sa["name"] if sa["ops"] > sb["ops"] else sb["name"]
        recent_leader = sa["name"] if (ra.get("ops") or 0) > (rb.get("ops") or 0) else sb["name"]
        form_split = season_leader != recent_leader

    sp_gap = abs(sp_h_score - sp_a_score)
    sp_split = sp_gap > 0.25  # meaningful SP advantage exists

    # sp_conflict: SP edge points against the win-prob pick — genuine disagreement
    sp_conflict = sp_split and (
        (sp_h_score > sp_a_score) != (pct_h > pct_a)
    )

    if (models_agree and prob_strength == "strong"
            and not form_split and not sp_conflict):
        level, emoji, color = "High",       "🟢", "#00c07a"
    elif models_agree and prob_strength in ("strong", "moderate") and not sp_conflict:
        level, emoji, color = "Moderate",   "🟡", "#f5c842"
    elif models_agree:
        level, emoji, color = "Low",        "🟠", "#f5a623"
    else:
        level, emoji, color = "Conflicted", "🔴", "#ff5252"

    reasons = []
    if not models_agree:
        reasons.append(f"Win probability favors **{prob_winner}**, but the projected run line "
                       f"favors **{margin_winner}** — the models disagree.")
    if sp_conflict:
        sp_fav_prob = prob_winner
        sp_fav_sp   = sa["name"] if sp_h_score > sp_a_score else sb["name"]
        reasons.append(f"Today's starting pitcher favors **{sp_fav_sp}** but the win probability "
                       f"model favors **{sp_fav_prob}** — the SP edge contradicts the model pick.")
    elif sp_split:
        sp_fav = sa["name"] if sp_h_score > sp_a_score else sb["name"]
        reasons.append(f"Today's starting pitcher gives **{sp_fav}** a meaningful edge "
                       f"(SP quality gap: {round(sp_gap, 2)}).")
    if form_split and ra and rb:
        season_leader = sa["name"] if sa["ops"] > sb["ops"] else sb["name"]
        recent_leader = sa["name"] if (ra.get("ops") or 0) > (rb.get("ops") or 0) else sb["name"]
        reasons.append(f"Recent form and season form diverge: **{recent_leader}** looks "
                       f"better recently but **{season_leader}** has the stronger season OPS.")
    if ops_rec_split:
        reasons.append(f"**{ops_leader}** has the higher OPS but **{rec_leader}** has the "
                       f"better record — schedule difficulty may be a factor.")
    if prob_strength == "narrow" and models_agree:
        reasons.append(f"Both models agree on **{prob_winner}** but the edge is slim "
                       f"({prob_gap}pp) — a different bullpen call could flip the result.")
    if not reasons:
        reasons.append(f"Both models consistently favor **{prob_winner}** with a "
                       f"{prob_strength} edge across offense, pitching, and record.")
    return level, emoji, color, reasons


def park_factor_reason(venue_name: str, park_factor: float) -> str | None:
    """Return a confidence reason string if the park is significantly hitter/pitcher-friendly."""
    if park_factor >= 1.05:
        return (f"**{venue_name}** is a hitter-friendly park (factor: {park_factor:.2f}x) "
                f"— expect elevated run totals and wider margins.")
    if park_factor <= 0.96:
        return (f"**{venue_name}** is a pitcher-friendly park (factor: {park_factor:.2f}x) "
                f"— run totals may be suppressed and games closer than expected.")
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚾ MLB Daily Slate")
    st.caption(f"Season: {SEASON} · statsapi · refreshes every 30 min")

    st.markdown("---")
    st.markdown("##### Settings")
    home_display = st.checkbox("Show home team advantage", value=True)
    ng = st.select_slider("Recent form window (days)", options=[7, 14, 21, 30], value=14)

    st.markdown("---")
    st.markdown("##### Form vs. Season balance")
    # Single source of truth: choose the recent-form weight; season is the remainder.
    # Previously two independent 0-100 sliders were normalized after the fact, so
    # setting season=65 while recent sat at its default 50 silently produced
    # 56.5/43.5 — not 65/35. One slider makes the ratio exact.
    w_recent_pct = st.slider(
        "Recent-form weight (%)", 0, 100, 35, step=5,
        help="Season weight is the remainder. 35 → 65% season / 35% recent.")
    w_season_pct = 100 - w_recent_pct
    w_season = w_season_pct / 100
    w_recent = w_recent_pct / 100
    st.caption(f"Blend: **{w_season_pct}% season / {w_recent_pct}% recent**")

    st.markdown("---")
    st.markdown("##### Model weights")
    w_off_raw = st.slider("Offense weight",  0, 100, 35, step=5)
    w_def_raw = st.slider("Pitching weight", 0, 100, 35, step=5)
    w_rec_raw = st.slider("Record weight",   0, 100, 30, step=5)
    total_w   = (w_off_raw + w_def_raw + w_rec_raw) or 1
    w_off = w_off_raw / total_w
    w_def = w_def_raw / total_w
    w_rec = w_rec_raw / total_w

    st.markdown("---")
    with st.spinner("Loading MLB stats..."):
        batting_data, pitching_data = fetch_team_stats(SEASON)
        standings_data = fetch_standings()
        # Enrich standings with real run differential computed from game scores
        rd_data = fetch_run_differential(SEASON)
        for full_name, rd_info in rd_data.items():
            if full_name in standings_data:
                standings_data[full_name]["RD"]    = rd_info["rd"]
                standings_data[full_name]["RD_PG"] = rd_info["rd_pg"]

    with st.spinner(f"Loading last {ng}-day form..."):
        batting_recent, pitching_recent = fetch_team_stats_recent(SEASON, ng)

    recent_populated = sum(
        1 for v in batting_recent.values()
        if float(v.get("G", 0) or 0) >= 1
    )
    if recent_populated == 0:
        st.warning("⚠️ Recent form data unavailable — using season stats only.", icon="📊")
    elif recent_populated < 20:
        st.info(f"📊 Recent form loaded for {recent_populated}/30 teams.")
    else:
        st.success(f"✅ Recent form loaded for {recent_populated}/30 teams.")

    # Auto-save today's snapshot including today's probable SP stats
    # Fetch today's games to get probable pitchers, then snapshot their stats
    try:
        today_str  = datetime.datetime.today().strftime("%m/%d/%Y")
        snap_games = statsapi.schedule(date=today_str, sportId=1)
        snap_games = [g for g in snap_games if g.get("game_type") == "R"]
        pitcher_ids = set()
        for sg in snap_games:
            for pid_key in ["home_pitcher_id", "away_pitcher_id"]:
                pid = sg.get(pid_key)
                if isinstance(pid, int) and pid:
                    pitcher_ids.add(pid)
            for name_key in ["home_probable_pitcher", "away_probable_pitcher"]:
                sp_name = sg.get(name_key, "")
                if sp_name and sp_name != "TBD":
                    try:
                        res = statsapi.lookup_player(sp_name)
                        if res:
                            pitcher_ids.add(res[0]["id"])
                    except Exception:
                        pass
        # Fetch stats for each pitcher
        todays_pitcher_stats = {}
        for pid in pitcher_ids:
            stats = fetch_pitcher_season_stats(pid, SEASON)
            if stats:
                todays_pitcher_stats[pid] = stats
    except Exception:
        todays_pitcher_stats = {}

    save_snapshot(batting_data, pitching_data, standings_data,
                  pitchers=todays_pitcher_stats,
                  batting_recent=batting_recent, pitching_recent=pitching_recent,
                  recent_window=ng)

    games_played = [v.get("G", 0) for v in batting_data.values()]
    avg_games    = sum(games_played) / len(games_played) if games_played else 0
    if avg_games < 5:
        st.warning(f"⚠️ Only ~{round(avg_games, 1)} games/team so far. "
                   f"Stats are very early and may be unreliable.", icon="⚾")

    st.caption(f"Stats loaded for {len(batting_data)}/30 teams · "
               f"avg {round(avg_games, 0):.0f} games played")


# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_today, tab_back = st.tabs(["⚾ Today's Games", "📊 Backtest"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Today's Games
# ═══════════════════════════════════════════════════════════════════════════════
with tab_today:
    today_label = datetime.datetime.today().strftime("%A, %B %d").replace(" 0", " ")
    st.header(f"Today's Slate — {today_label}")
    st.caption("Every regular season game today run through the model, ranked by confidence. "
               "Starting pitcher stats are incorporated into each prediction.")

    with st.spinner("Fetching today's schedule..."):
        todays_games, todays_err = fetch_todays_games()

    if todays_err:
        st.error(f"Could not load schedule: {todays_err}")
        st.stop()

    if not todays_games:
        st.info("No MLB regular season games scheduled for today. Check back tomorrow!")
        st.stop()

    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_pitcher_stats_by_id(player_id: int, season: int) -> dict:
        """Fetch pitcher stats directly by ID — faster and more reliable than name lookup."""
        if not player_id:
            return {}
        return fetch_pitcher_season_stats(player_id, season)


    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_pitcher_stats_by_name(name: str, season: int) -> dict:
        """Fallback name-based lookup for when ID is not available."""
        if not name or name == "TBD":
            return {}
        try:
            res = statsapi.lookup_player(name)
            if not res:
                return {}
            return fetch_pitcher_season_stats(res[0]["id"], season)
        except Exception:
            return {}


    # ── Score each game ────────────────────────────────────────────────────────────
    slate_results = []
    skipped       = []

    for game in todays_games:
        h_name = game.get("home_name", "")
        a_name = game.get("away_name", "")
        h_abb  = api_name_to_abb(h_name)
        a_abb  = api_name_to_abb(a_name)
        if not h_abb or not a_abb:
            skipped.append(f"{a_name} @ {h_name}")
            continue

        s_h = score_team(h_abb, batting_data,   pitching_data,   standings_data)
        s_a = score_team(a_abb, batting_data,   pitching_data,   standings_data)
        r_h = score_team(h_abb, batting_recent, pitching_recent, standings_data) if batting_recent else None
        r_a = score_team(a_abb, batting_recent, pitching_recent, standings_data) if batting_recent else None

        # Verify blend is actually happening — check if recent data has valid G
        h_recent_valid = bool(r_h and float(r_h.get("G", 0) or 0) >= 1)
        a_recent_valid = bool(r_a and float(r_a.get("G", 0) or 0) >= 1)

        # Fetch starting pitcher stats — prefer ID-based lookup (more reliable)
        home_sp_name = game.get("home_probable_pitcher", "TBD")
        away_sp_name = game.get("away_probable_pitcher", "TBD")
        home_sp_id   = game.get("home_pitcher_id") or game.get("home_pitcher_note")
        away_sp_id   = game.get("away_pitcher_id") or game.get("away_pitcher_note")

        # Use numeric ID if available, otherwise fall back to name lookup
        if isinstance(home_sp_id, int) and home_sp_id:
            home_sp_stats = fetch_pitcher_stats_by_id(home_sp_id, SEASON)
        else:
            home_sp_stats = fetch_pitcher_stats_by_name(home_sp_name, SEASON)

        if isinstance(away_sp_id, int) and away_sp_id:
            away_sp_stats = fetch_pitcher_stats_by_id(away_sp_id, SEASON)
        else:
            away_sp_stats = fetch_pitcher_stats_by_name(away_sp_name, SEASON)

        sp_h_score = pitcher_adjustment(home_sp_stats)
        sp_a_score = pitcher_adjustment(away_sp_stats)

        # Ballpark factor for this game's venue
        venue_id   = game.get("venue_id")
        park_factor = ballpark_factor(venue_id, h_abb)

        home_flag = "home" if home_display else "neutral"

        ph, pa = calc_prob(s_h, s_a, home_flag, w_off, w_def, w_rec,
                           r_h, r_a, w_season, w_recent, sp_h_score, sp_a_score,
                           park_factor=park_factor)
        prob_pick = s_h["name"] if ph >= pa else s_a["name"]
        home_pct  = round(ph * 100)
        away_pct  = round(pa * 100)

        proj_h, proj_a, proj_margin, margin_pick, raw_margin, proj_total = calc_run_line(
            s_h, s_a, home_flag, w_off, w_def, w_rec,
            r_h, r_a, w_season, w_recent, sp_h_score, sp_a_score,
            park_factor=park_factor
        )

        conf_l, conf_e, conf_color, conf_reasons = calc_confidence(
            s_h, s_a, home_pct, away_pct, margin_pick, prob_pick,
            r_h, r_a, sp_h_score, sp_a_score
        )
        # Append park note if venue is meaningfully extreme
        park_note = park_factor_reason(game.get("venue_name", "this park"), park_factor)
        if park_note:
            conf_reasons.append(park_note)

        # Game time (convert UTC to local display)
        game_time_utc = game.get("game_datetime", "")
        try:
            gt = datetime.datetime.strptime(game_time_utc, "%Y-%m-%dT%H:%M:%SZ")
            game_time_str = gt.strftime("%I:%M %p UTC").lstrip("0")
        except Exception:
            game_time_str = game.get("game_time", "")

        status    = game.get("status", "")
        completed = status == "Final"
        winner_name, cover_result = None, None
        if completed:
            hs  = int(game.get("home_score", 0) or 0)
            as_ = int(game.get("away_score", 0) or 0)
            winner_name   = s_h["name"] if hs > as_ else s_a["name"]
            actual_margin = abs(hs - as_)
            actual_total = hs + as_
            if margin_pick == winner_name:
                # Standard MLB run line: favorite must win by 2+ (covers -1.5)
                did_cover_today = actual_margin >= 2
                cover_result = (f"✅ Covered -1.5 (won by {actual_margin})"
                                if did_cover_today
                                else f"❌ No cover -1.5 (won by {actual_margin})")
            else:
                did_cover_today = False
                cover_result = "❌ Lost outright"

        slate_results.append({
            "home":          s_h["name"],
            "away":          s_a["name"],
            "home_record":   f"{s_h['w']}–{s_h['l']}",
            "away_record":   f"{s_a['w']}–{s_a['l']}",
            "home_sp":       home_sp_name,
            "away_sp":       away_sp_name,
            "home_sp_stats": home_sp_stats,
            "away_sp_stats": away_sp_stats,
            "sp_h_score":    sp_h_score,
            "sp_a_score":    sp_a_score,
            "prob_pick":     prob_pick,
            "home_pct":      home_pct,
            "away_pct":      away_pct,
            "margin_pick":   margin_pick,
            "proj_home":     proj_h,
            "proj_away":     proj_a,
            "proj_margin":   proj_margin,
            "conf_level":    conf_l,
            "conf_emoji":    conf_e,
            "conf_color":    conf_color,
            "conf_reasons":  conf_reasons,
            "game_time":     game_time_str,
            "status":        status,
            "completed":     completed,
            "winner":        winner_name,
            "cover_result":  cover_result,
            "park_factor":     park_factor,
            "venue_name":      game.get("venue_name", ""),
            "proj_total":      proj_total,
            "h_recent_valid":  h_recent_valid,
            "a_recent_valid":  a_recent_valid,
            # key team stats for display
            "home_ops":  s_h["ops"],  "away_ops":  s_a["ops"],
            "home_era":  s_h["era"],  "away_era":  s_a["era"],
            "home_fip":  s_h["fip"],  "away_fip":  s_a["fip"],
            "home_rd":   s_h["rd_pg"],"away_rd":   s_a["rd_pg"],
            "home_wpct": s_h["wpct"], "away_wpct": s_a["wpct"],
        })

    # Sort by confidence then edge size
    tier_rank = {"High": 0, "Moderate": 1, "Low": 2, "Conflicted": 3}
    slate_results.sort(key=lambda x: (tier_rank.get(x["conf_level"], 9),
                                       -abs(x["home_pct"] - 50)))

    if skipped:
        st.warning(f"⚠️ {len(skipped)} game(s) skipped (team name not recognized): "
                   f"{', '.join(skipped)}")

    st.caption(f"{len(slate_results)} games · sorted by confidence then edge size · "
               f"season/recent {round(w_season*100)}/{round(w_recent*100)} · "
               f"offense/pitching/record {round(w_off*100)}/{round(w_def*100)}/{round(w_rec*100)}")
    st.markdown("")

    # ── HTML helper functions (defined once, used in render loop) ─────────────────
    def _sp_line(stats: dict, name: str) -> str:
        if not stats:
            return f"<span style='color:#666'>{name} &middot; No stats yet</span>"
        fip_str = f"{stats.get('fip','—')}" if stats.get('fip') else '—'
        bb9_str = f"{stats.get('bb9','—')}" if stats.get('bb9') else '—'
        gs_str  = f"{stats.get('gs', 0)} GS" if stats.get('gs') else ''
        return (f"<span style='color:#ccc'>{name}</span> "
                f"<span style='color:#888;font-size:11px;'>"
                f"ERA {stats.get('era','—')} &middot; "
                f"FIP {fip_str} &middot; "
                f"WHIP {stats.get('whip','—')} &middot; "
                f"K/9 {stats.get('k9','—')} &middot; "
                f"BB/9 {bb9_str} &middot; "
                f"{stats.get('wins',0)}W-{stats.get('losses',0)}L {gs_str}"
                f"</span>")


    def _sp_bar(score: float) -> str:
        pct   = int(score * 100)
        col   = "#00c07a" if score > 0.6 else ("#f5c842" if score > 0.4 else "#ff5252")
        label = "Elite" if score > 0.7 else ("Good" if score > 0.55 else
                ("Avg" if score > 0.4 else "Below Avg"))
        return (f"<div style='display:flex;align-items:center;gap:6px;margin-top:3px;'>"
                f"<div style='flex:1;height:4px;background:#333;border-radius:2px;'>"
                f"<div style='width:{pct}%;height:4px;background:{col};border-radius:2px;'>"
                f"</div></div>"
                f"<span style='font-size:10px;color:{col};'>{label}</span></div>")


    # ── Render each game ───────────────────────────────────────────────────────────
    for game in slate_results:
        color = game["conf_color"]

        # Status badge
        if game["completed"]:
            status_badge = (
                f"<span style='background:rgba(0,192,122,0.15);color:#00c07a;font-size:11px;"
                f"padding:2px 8px;border-radius:4px;'>Final · {game['winner']} won</span>"
            )
        elif game["status"] in ("In Progress", "Live"):
            status_badge = (
                "<span style='background:rgba(255,82,82,0.15);color:#ff5252;font-size:11px;"
                "padding:2px 8px;border-radius:4px;'>🔴 Live</span>"
            )
        else:
            status_badge = (
                f"<span style='background:rgba(61,139,255,0.15);color:#3d8bff;font-size:11px;"
                f"padding:2px 8px;border-radius:4px;'>{game['game_time']}</span>"
            )

        cover_badge = ""
        if game["completed"] and game["cover_result"]:
            cover_badge = (f"<span style='font-size:11px;color:#aaa;margin-left:6px;'>"
                           f"{game['cover_result']}</span>")

        home_sp_html   = _sp_line(game["home_sp_stats"], game["home_sp"])
        away_sp_html   = _sp_line(game["away_sp_stats"], game["away_sp"])
        home_bar_html  = _sp_bar(game["sp_h_score"])
        away_bar_html  = _sp_bar(game["sp_a_score"])
        reasons_html   = "".join(
            f"<div style='margin:2px 0;font-size:12px;color:#aaa;'>&bull; {r}</div>"
            for r in game["conf_reasons"]
        )

        # Pre-compute all conditional values to avoid quote conflicts in f-string
        home_prob_color  = "#00c07a" if game["prob_pick"] == game["home"] else "#aaa"
        home_prob_weight = "800"     if game["prob_pick"] == game["home"] else "400"
        away_prob_color  = "#00c07a" if game["prob_pick"] == game["away"] else "#aaa"
        away_prob_weight = "800"     if game["prob_pick"] == game["away"] else "400"

        home_name    = game["home"]
        away_name    = game["away"]
        home_record  = game["home_record"]
        away_record  = game["away_record"]

        # Recent form blend indicator — shows whether slider is actually working
        h_tag = "🟢 recent" if game.get("h_recent_valid") else "⚪ season only"
        a_tag = "🟢 recent" if game.get("a_recent_valid") else "⚪ season only"
        blend_note = (
            f"<span style='font-size:10px;color:#666;'>"
            f"Data: {home_name.split()[-1]} {h_tag} · "
            f"{away_name.split()[-1]} {a_tag}"
            f"</span>"
        )
        home_pct     = game["home_pct"]
        away_pct     = game["away_pct"]
        proj_home    = game["proj_home"]
        proj_away    = game["proj_away"]
        margin_pick  = game["margin_pick"]
        proj_margin  = game["proj_margin"]
        home_ops     = game["home_ops"]
        away_ops     = game["away_ops"]
        home_era     = game["home_era"]
        away_era     = game["away_era"]
        home_wpct    = game["home_wpct"]
        away_wpct    = game["away_wpct"]
        prob_pick    = game["prob_pick"]
        conf_emoji   = game["conf_emoji"]
        conf_level   = game["conf_level"]
        home_sp_name = game["home_sp"]
        away_sp_name = game["away_sp"]

        card_html = (
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;'
            f'border-left:4px solid {color};border-radius:12px;padding:18px 22px;margin-bottom:16px;">'

            # Header
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'flex-wrap:wrap;gap:8px;margin-bottom:14px;">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
            f'<span style="font-size:17px;font-weight:700;">'
            f'{away_name} <span style="color:#555;font-size:13px;font-weight:400;">({away_record})</span>'
            f' <span style="color:#555;margin:0 6px;">@</span> '
            f'{home_name} <span style="color:#555;font-size:13px;font-weight:400;">({home_record})</span>'
            f'</span> {status_badge} {cover_badge}</div>'
            f'<div style="margin-top:4px;">{blend_note}</div>'
            f'<span style="font-size:13px;font-weight:700;color:{color};">'
            f'{conf_emoji} {conf_level} confidence</span></div>'

            # Stats grid
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:14px;">'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Win probability</div>'
            f'<div style="font-size:14px;">'
            f'<span style="color:{home_prob_color};font-weight:{home_prob_weight};">'
            f'{home_name} {home_pct}%</span><br>'
            f'<span style="color:{away_prob_color};font-weight:{away_prob_weight};">'
            f'{away_name} {away_pct}%</span></div></div>'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Projected score</div>'
            f'<div style="font-size:14px;font-weight:600;">'
            f'{home_name} <span style="color:#f5c842;">{proj_home}</span><br>'
            f'{away_name} <span style="color:#f5c842;">{proj_away}</span><br>'
            f'<span style="font-size:11px;color:#888;font-weight:400;">O/U: '
            f'<span style="color:#f5c842;">{game["proj_total"]}</span></span></div></div>'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Run line (-1.5)</div>'
            f'<div style="font-size:14px;font-weight:700;color:{color};">'
            f'{margin_pick} -1.5'
            f'<br><span style="font-size:11px;font-weight:400;color:#888;">'
            f'Proj margin: {proj_margin} runs</span></div></div>'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Key stats</div>'
            f'<div style="font-size:11px;color:#aaa;">'
            f'OPS: <span style="color:#ccc;">{home_ops} / {away_ops}</span><br>'
            f'ERA/FIP: <span style="color:#ccc;">{home_era}/{game["home_fip"]} / {away_era}/{game["away_fip"]}</span><br>'
            f'RD/G: <span style="color:#ccc;">{game["home_rd"]:+.2f} / {game["away_rd"]:+.2f}</span><br>'
            f'W%: <span style="color:#ccc;">{home_wpct}% / {away_wpct}%</span><br>'
            f'Park: <span style="color:{"#f5a623" if abs(game["park_factor"]-1)>0.03 else "#ccc"};">'
            f'{game["venue_name"]} ({game["park_factor"]:+.0%})</span>'
            f'</div></div></div>'

            # Starting pitchers
            f'<div style="border-top:1px solid rgba(255,255,255,0.07);padding-top:12px;margin-bottom:10px;">'
            f'<div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:8px;">Starting pitchers</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">'
            f'<div><div style="font-size:10px;color:#888;margin-bottom:2px;">{home_name} (home)</div>'
            f'{home_sp_html}{home_bar_html}</div>'
            f'<div><div style="font-size:10px;color:#888;margin-bottom:2px;">{away_name} (away)</div>'
            f'{away_sp_html}{away_bar_html}</div>'
            f'</div></div>'

            # Reasons
            f'<div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:10px;">'
            f'{reasons_html}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Backtest
# ═══════════════════════════════════════════════════════════════════════════════
with tab_back:
    st.header(f"Model Backtesting — {SEASON} Season")

    available_snaps = list_snapshots()
    snap_count = len(available_snaps)
    if snap_count == 0:
        st.caption(
            "⚠️ No snapshots saved yet — backtesting is using current stats (look-ahead bias). "
            "The app will save a snapshot each day it runs. Come back tomorrow for clean backtesting."
        )
    elif snap_count < 7:
        st.caption(
            f"📸 {snap_count} daily snapshot(s) saved. Games with a matching snapshot use "
            f"pre-game stats. Games before {available_snaps[0]} still use current stats."
        )
    else:
        st.caption(
            f"📸 {snap_count} daily snapshots available ({available_snaps[0]} → {available_snaps[-1]}). "
            f"All games with a prior-day snapshot use pre-game stats — no look-ahead bias."
        )

    num_games = st.slider("Games to evaluate (most recent first)", 10, 200, 100, step=10)

    with st.spinner("Fetching season results..."):
        season_games, season_err = fetch_season_games(SEASON)

    if season_err:
        st.error(f"Could not load season games: {season_err}")
    elif not season_games:
        st.info("No completed regular season games found yet this season.")
    else:
        results_table = []
        correct_prob = covered = total = 0
        margin_errors = []

        snap_hits = snap_misses = 0
        recent_used = 0   # games whose matched snapshot carried recent form (true 65/35)
        # Pre-load all snapshots into memory once — avoids repeated disk reads per game
        snap_cache = {}
        snap_cache_pitchers = {}   # {snap_date: {pid_str: stats}}
        snap_cache_recent = {}     # {snap_date: (batting_recent, pitching_recent)}
        for snap_date in list_snapshots():
            b, p, s = load_snapshot(snap_date)
            if b and p:
                snap_cache[snap_date] = (b, p, s)
            sp_data = load_sp_snapshot(snap_date)
            if sp_data:
                snap_cache_pitchers[snap_date] = sp_data
            rb, rp = load_recent_snapshot(snap_date)
            if rb:
                snap_cache_recent[snap_date] = (rb, rp)

        def get_snapshot_from_cache(game_date_str: str):
            """Find the most recent snapshot before game_date using the in-memory cache."""
            try:
                game_dt = datetime.datetime.strptime(game_date_str, "%Y-%m-%d")
            except ValueError:
                return None, None, None, None
            for snap_date in sorted(snap_cache.keys(), reverse=True):
                snap_dt = datetime.datetime.strptime(snap_date, "%Y-%m-%d")
                if snap_dt < game_dt:
                    b, p, s = snap_cache[snap_date]
                    return b, p, s, snap_date
            return None, None, None, None

        for game in sorted(season_games, key=lambda g: g.get("game_date", ""), reverse=True)[:num_games]:
            h_abb = api_name_to_abb(game.get("home_name", ""))
            a_abb = api_name_to_abb(game.get("away_name", ""))
            if not h_abb or not a_abb:
                continue
            hs = int(game.get("home_score", 0) or 0)
            as_ = int(game.get("away_score", 0) or 0)
            game_date = game.get("game_date", "")[:10]
            try:
                # Try to find a snapshot taken before this game was played
                snap_bat, snap_pit, snap_std, snap_date = get_snapshot_from_cache(game_date)
                if snap_bat and snap_pit:
                    bt_batting   = snap_bat
                    bt_pitching  = snap_pit
                    bt_standings = snap_std or standings_data
                    used_snapshot = snap_date
                    snap_hits += 1
                else:
                    # No snapshot available — fall back to current stats
                    bt_batting   = batting_data
                    bt_pitching  = pitching_data
                    bt_standings = standings_data
                    used_snapshot = None
                    snap_misses += 1
                s_h = score_team(h_abb, bt_batting, bt_pitching, bt_standings)
                s_a = score_team(a_abb, bt_batting, bt_pitching, bt_standings)

                # Recent form, as snapshotted BEFORE this game (no look-ahead).
                # Only available when the matched snapshot carried recent form;
                # older snapshots leave this empty and the game runs season-only.
                bt_rec = snap_cache_recent.get(used_snapshot, (None, None)) if used_snapshot else (None, None)
                bt_bat_recent, bt_pit_recent = bt_rec
                bt_r_h = (score_team(h_abb, bt_bat_recent, bt_pit_recent, bt_standings)
                          if bt_bat_recent else None)
                bt_r_a = (score_team(a_abb, bt_bat_recent, bt_pit_recent, bt_standings)
                          if bt_bat_recent else None)
                if bt_r_h and bt_r_a:
                    recent_used += 1

                # Get SP stats from snapshot if available, fall back to live lookup
                bt_sp_pitchers = snap_cache_pitchers.get(used_snapshot, {}) if used_snapshot else {}
                bt_home_sp_id  = game.get("home_pitcher_id")
                bt_away_sp_id  = game.get("away_pitcher_id")
                bt_home_sp     = get_sp_stats_from_cache(bt_sp_pitchers, bt_home_sp_id)
                bt_away_sp     = get_sp_stats_from_cache(bt_sp_pitchers, bt_away_sp_id)
                # Fall back to name lookup only if no snapshot SP data
                if not bt_home_sp:
                    sp_name = game.get("home_probable_pitcher", "")
                    if sp_name and sp_name != "TBD":
                        bt_home_sp = fetch_pitcher_stats_by_name(sp_name, SEASON)
                if not bt_away_sp:
                    sp_name = game.get("away_probable_pitcher", "")
                    if sp_name and sp_name != "TBD":
                        bt_away_sp = fetch_pitcher_stats_by_name(sp_name, SEASON)
                bt_sp_h_score = pitcher_adjustment(bt_home_sp)
                bt_sp_a_score = pitcher_adjustment(bt_away_sp)
                bt_park_factor = PARK_RUN_FACTOR.get(h_abb, 1.0)

                ph, pa    = calc_prob(s_h, s_a, "home", w_off, w_def, w_rec,
                                      bt_r_h, bt_r_a, w_season, w_recent,
                                      sp_h_score=bt_sp_h_score, sp_a_score=bt_sp_a_score,
                                      park_factor=bt_park_factor)
                prob_pick = s_h["name"] if ph >= pa else s_a["name"]
                prob_pct  = round(max(ph, pa) * 100)
                _, _, proj_margin, margin_pick, _, proj_total_bt = calc_run_line(
                    s_h, s_a, "home", w_off, w_def, w_rec,
                    bt_r_h, bt_r_a, w_season, w_recent,
                    sp_h_score=bt_sp_h_score, sp_a_score=bt_sp_a_score)
                conf_l, conf_e, _, _ = calc_confidence(
                    s_h, s_a, round(ph*100), round(pa*100), margin_pick, prob_pick,
                    bt_r_h, bt_r_a, sp_h_score=bt_sp_h_score, sp_a_score=bt_sp_a_score)
                actual_winner = s_h["name"] if hs > as_ else s_a["name"]
                actual_margin = round(abs(hs - as_), 1)
                if margin_pick == actual_winner:
                    # Standard MLB run line: favorite must win by 2+ (covers -1.5)
                    did_cover = actual_margin >= 2
                    cover_str = (f"✅ -1.5 (won by {actual_margin})"
                                 if did_cover else f"❌ -1.5 (won by {actual_margin})")
                else:
                    did_cover = False
                    cover_str = "❌ Lost outright"
                prob_correct = prob_pick == actual_winner
                margin_err   = round(abs(proj_margin - actual_margin), 1)
                if prob_correct: correct_prob += 1
                if did_cover:    covered += 1
                margin_errors.append(margin_err)
                total += 1
                results_table.append({
                    "date":         game_date,
                    "matchup":      f"{s_a['name']} @ {s_h['name']}",
                    "used_snapshot": used_snapshot,
                    "actual":       f"{actual_winner} ({max(hs,as_)}–{min(hs,as_)})",
                    "actual_margin":actual_margin,
                    "prob_pick":    prob_pick,
                    "prob_pct":     prob_pct,
                    "prob_correct": prob_correct,
                    "margin_pick":  margin_pick,
                    "proj_margin":  proj_margin,
                    "did_cover":    did_cover,
                    "cover_str":    cover_str,
                    "margin_err":   margin_err,
                    "confidence":   f"{conf_e} {conf_l}",
                    "conf_level":   conf_l,
                })
            except Exception:
                continue

        if total > 0:
            acc_prob       = round(correct_prob / total * 100)
            cover_rate     = round(covered / total * 100)
            avg_margin_err = round(sum(margin_errors) / len(margin_errors), 1) if margin_errors else 0

            # Show snapshot coverage
            if snap_hits + snap_misses > 0:
                snap_pct = round(snap_hits / (snap_hits + snap_misses) * 100)
                oldest_snap = available_snaps[0] if available_snaps else None
                if snap_pct == 100:
                    st.success(
                        f"✅ All {total} games evaluated using pre-game snapshots — no look-ahead bias."
                    )
                elif snap_pct > 0:
                    st.info(
                        f"📸 {snap_hits}/{snap_hits+snap_misses} games ({snap_pct}%) used pre-game snapshots. "
                        f"{snap_misses} earlier games used current stats — these predate your first snapshot "
                        f"({oldest_snap}). Look-ahead bias only applies to those older games."
                    )
                else:
                    # snap_hits == 0 but snapshots exist — all games predate the snapshots
                    if available_snaps:
                        st.info(
                            f"📸 You have {len(available_snaps)} snapshot(s) saved "
                            f"(from {available_snaps[0]} onward), but all backtested games "
                            f"predate your earliest snapshot. Snapshots will be used for new games "
                            f"going forward. Current stats used for all historical games."
                        )
                    else:
                        st.warning(
                            "⚠️ No snapshots saved yet — all games evaluated using current stats. "
                            "The app saves a snapshot each day it runs."
                        )

            # Show how many games actually exercised the recent-form blend.
            # Snapshots saved before recent-form snapshotting was added carry no
            # recent stats, so those games run season-only even at w_recent > 0.
            if w_recent > 0 and total > 0:
                rec_pct = round(recent_used / total * 100)
                if recent_used == 0:
                    st.warning(
                        f"⚠️ Recent-form weight is {round(w_recent*100)}%, but **0 of {total}** "
                        f"backtested games carried snapshotted recent form — they all ran "
                        f"season-only. Your existing snapshots predate recent-form capture, so "
                        f"this backtest does NOT yet reflect the {round(w_season*100)}/{round(w_recent*100)} "
                        f"blend. New snapshots (saved each day going forward) will include it."
                    )
                elif rec_pct < 100:
                    st.info(
                        f"📊 {recent_used}/{total} games ({rec_pct}%) used the "
                        f"{round(w_season*100)}/{round(w_recent*100)} season/recent blend; the rest ran "
                        f"season-only (their snapshots predate recent-form capture)."
                    )
                else:
                    st.success(
                        f"✅ All {total} games evaluated with the "
                        f"{round(w_season*100)}/{round(w_recent*100)} season/recent blend you run live."
                    )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games evaluated", total)
            m2.metric("Win prob accuracy", f"{acc_prob}%", f"{correct_prob}/{total} correct")
            m3.metric("Run line cover rate", f"{cover_rate}%", f"{covered}/{total} covered")
            m4.metric("Avg run-line error", f"{avg_margin_err} runs")

            # Accuracy by confidence tier
            st.markdown('<p class="section-head">Accuracy by confidence tier</p>', unsafe_allow_html=True)
            tier_order  = ["High", "Moderate", "Low", "Conflicted"]
            tier_colors = {"High": "#00c07a", "Moderate": "#f5c842", "Low": "#f5a623", "Conflicted": "#ff5252"}
            tier_emoji  = {"High": "🟢", "Moderate": "🟡", "Low": "🟠", "Conflicted": "🔴"}
            tier_stats  = {t: {"prob_hit": 0, "covered": 0, "total": 0, "errs": []} for t in tier_order}
            for row in results_table:
                lvl = row["conf_level"]
                if lvl in tier_stats:
                    tier_stats[lvl]["total"] += 1
                    if row["prob_correct"]: tier_stats[lvl]["prob_hit"] += 1
                    if row["did_cover"]:    tier_stats[lvl]["covered"]  += 1
                    tier_stats[lvl]["errs"].append(row["margin_err"])

            active_tiers = [t for t in tier_order if tier_stats[t]["total"] > 0]
            if active_tiers:
                cols = st.columns(len(active_tiers))
                for i, tier in enumerate(active_tiers):
                    ts      = tier_stats[tier]
                    pp      = round(ts["prob_hit"] / ts["total"] * 100) if ts["total"] else 0
                    cp      = round(ts["covered"]  / ts["total"] * 100) if ts["total"] else 0
                    ae      = round(sum(ts["errs"]) / len(ts["errs"]), 1) if ts["errs"] else 0
                    c       = tier_colors[tier]
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.04);border:1px solid {c}44;
                                    border-left:4px solid {c};border-radius:10px;padding:14px 16px;">
                            <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                                        color:{c};margin-bottom:4px;">{tier_emoji[tier]} {tier}</div>
                            <div style="font-size:11px;color:#888;margin-bottom:10px;">{ts['total']} games</div>
                            <div style="font-size:13px;margin-bottom:4px;">
                                <span style="color:#888;">Win prob </span>
                                <span style="font-weight:800;font-size:18px;
                                    color:{'#00c07a' if pp>=50 else '#ff5252'}">{pp}%</span>
                                <span style="color:#666;font-size:11px;"> ({ts['prob_hit']}/{ts['total']})</span>
                            </div>
                            <div style="font-size:13px;margin-bottom:4px;">
                                <span style="color:#888;">Covered </span>
                                <span style="font-weight:800;font-size:18px;
                                    color:{'#00c07a' if cp>=50 else '#ff5252'}">{cp}%</span>
                                <span style="color:#666;font-size:11px;"> ({ts['covered']}/{ts['total']})</span>
                            </div>
                            <div style="font-size:11px;color:#888;">Avg error: <span style="color:#ccc;">{ae} runs</span></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("")
                # Bar chart
                tier_labels = [f"{tier_emoji[t]} {t} ({tier_stats[t]['total']}g)" for t in active_tiers]
                prob_accs   = [round(tier_stats[t]["prob_hit"]/tier_stats[t]["total"]*100) for t in active_tiers]
                cov_rates   = [round(tier_stats[t]["covered"] /tier_stats[t]["total"]*100) for t in active_tiers]
                fig_tier = go.Figure()
                fig_tier.add_trace(go.Bar(name="Win prob accuracy", x=tier_labels, y=prob_accs,
                    marker_color="#00c07a", opacity=0.85,
                    text=[f"{v}%" for v in prob_accs], textposition="outside",
                    textfont=dict(color="#ccc", size=11)))
                fig_tier.add_trace(go.Bar(name="Run line cover rate", x=tier_labels, y=cov_rates,
                    marker_color="#3d8bff", opacity=0.85,
                    text=[f"{v}%" for v in cov_rates], textposition="outside",
                    textfont=dict(color="#ccc", size=11)))
                fig_tier.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                    annotation_text="50% baseline", annotation_font_color="rgba(255,255,255,0.4)",
                    annotation_position="right")
                fig_tier.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="group", height=280, margin=dict(l=10,r=10,t=30,b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
                    yaxis=dict(range=[0,120], ticksuffix="%", tickfont=dict(color="#888", size=10),
                               showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    xaxis=dict(tickfont=dict(color="#ccc", size=11)),
                )
                st.plotly_chart(fig_tier, use_container_width=True)

            # Overall accuracy bar
            st.markdown('<p class="section-head">Overall accuracy</p>', unsafe_allow_html=True)
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=["Win Probability Model", "Run Line Cover Rate"], y=[acc_prob, cover_rate],
                marker_color=["#00c07a" if acc_prob>=50 else "#ff5252",
                              "#3d8bff" if cover_rate>=50 else "#ff5252"],
                text=[f"{acc_prob}%", f"{cover_rate}%"],
                textposition="outside", textfont=dict(color="#ccc", size=13),
            ))
            fig_acc.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                annotation_text="50% baseline", annotation_font_color="rgba(255,255,255,0.4)",
                annotation_position="right")
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=10,r=10,t=30,b=10), showlegend=False,
                yaxis=dict(range=[0,120], ticksuffix="%", tickfont=dict(color="#888", size=10),
                           showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                xaxis=dict(tickfont=dict(color="#ccc", size=12)),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # Game-by-game table
            st.markdown('<p class="section-head">Most recent 25 games</p>', unsafe_allow_html=True)
            st.caption(f"Showing 25 of {total} games. Accuracy metrics above reflect all evaluated games.")
            for row in results_table[:25]:
                with st.container():
                    c1, c2, c3, c4, c5 = st.columns([1.2, 2.2, 1.8, 2.2, 1.2])
                    with c1:
                        snap_icon = "📸" if row.get("used_snapshot") else "⚠️"
                        snap_tip  = row["used_snapshot"] if row.get("used_snapshot") else "current stats"
                        st.caption(f"{row['date']} {snap_icon}")
                        st.caption(f"stats: {snap_tip}")
                    with c2:
                        st.markdown(f"**{row['matchup']}**")
                        st.caption(f"Result: {row['actual']}")
                    with c3:
                        icon = "✅" if row["prob_correct"] else "❌"
                        st.markdown(f"{icon} **{row['prob_pick']}**")
                        st.caption(f"Edge: {row['prob_pct']}%")
                    with c4:
                        st.markdown(row["cover_str"])
                        st.caption(f"RL: {row['margin_pick']} -1.5 · Proj margin: {row['proj_margin']} · Actual: {row['actual_margin']} runs")
                    with c5:
                        st.markdown(row["confidence"])
                    st.divider()
        else:
            st.info("Not enough data to evaluate yet.")

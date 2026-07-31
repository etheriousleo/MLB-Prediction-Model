"""
WNBA Win Probability — Today's Slate
--------------------------------------
Install dependencies:
    pip install requests streamlit

Run:
    streamlit run wnba_app.py

Data source: ESPN public API (no API key required)
"""

import datetime
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st

# ── Daily stat snapshots ────────────────────────────────────────────────────────
# The app archives one stats snapshot per day. Nothing in the app reads these
# anymore (the backtest tab is gone), but the write is kept because pre-game
# snapshots cannot be recreated after the fact — if a backtest is ever rebuilt,
# this archive is the only way it can score games with as-of-that-day stats
# instead of look-ahead data. Delete this block + the save_snapshot call in the
# sidebar if you never want a backtest again.
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wnba_snapshots")

def save_snapshot(team_stats: dict, standings: dict) -> None:
    """Write today's stats to disk once per day. Silent no-op on any failure."""
    date_str = datetime.datetime.today().strftime("%Y-%m-%d")
    path = os.path.join(SNAPSHOT_DIR, f"stats_{date_str}.json")
    if os.path.exists(path):
        return  # Already saved for today
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"date": date_str, "team_stats": team_stats,
                       "standings": standings}, f)
    except Exception:
        pass


# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="WNBA Daily Slate", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    .section-head {
        font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
        color: #888; margin: 1.5rem 0 0.5rem;
    }
    .stat-better { color: #f60; font-weight: 600; }
    .stat-worse  { color: #ff5252; }
    .confidence-reason { font-size: 0.85rem; line-height: 1.6; color: #ccc; }
</style>
""", unsafe_allow_html=True)

# Home-court advantage, in composite-score units. Under the logistic win-prob
# mapping (LOGISTIC_K = 6.0, defined with calc_prob), the slope near an even
# matchup is K/4 = 1.5, so 0.035 shifts the home team's win probability by
# ~1.5 * 0.035 ≈ +5pp — i.e. home wins ~55% of even matchups, consistent with
# historical WNBA home win rates and with HOME_PTS = 2.5 in the spread model.
# (SEASON is set by the sidebar selectbox; the old module-level SEASON here was
# dead code that the selectbox silently shadowed.)
HOME_BOOST = 0.035
HOME_PTS   = 2.5     # Expected home-court points bonus (spread model)

# ── Team lookup tables ───────────────────────────────────────────────────────────
# ESPN team IDs → our canonical abbreviations (confirmed May 2026)
ESPN_ID_TO_ABB = {
    "20":     "ATL",
    "19":     "CHI",
    "18":     "CON",
    "3":      "DAL",
    "129689": "GS",
    "5":      "IND",
    "17":     "LVA",
    "6":      "LA",
    "8":      "MIN",
    "9":      "NYL",
    "11":     "PHX",
    "132052": "PDX",
    "131935": "TOR",
    "16":     "WAS",
    # NOTE: Seattle previously shared id "16" here; a duplicate dict key is
    # silently overwritten by Python, so id 16 always mapped to WAS anyway.
    # SEA resolves through ESPN_ABB_TO_ABB instead — behavior unchanged.
}

# ESPN raw abbreviation → our canonical abbreviation
ESPN_ABB_TO_ABB = {
    "ATL":        "ATL",
    "CHI":        "CHI",
    "CONNECTICU": "CON",
    "CT":         "CON",
    "CON":        "CON",
    "DALLAS":     "DAL",
    "DAL":        "DAL",
    "GS":         "GS",
    "IND":        "IND",
    "LV":         "LVA",
    "LVA":        "LVA",
    "LA":         "LA",
    "MIN":        "MIN",
    "NY":         "NYL",
    "NYL":        "NYL",
    "PHX":        "PHX",
    "POR":        "PDX",
    "PDX":        "PDX",
    "SEA":        "SEA",
    "TOR":        "TOR",
    "WSH":        "WAS",
    "WAS":        "WAS",
}

ABB_TO_FULL = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "GS":  "Golden State Valkyries",
    "IND": "Indiana Fever",
    "LA":  "Los Angeles Sparks",
    "LVA": "Las Vegas Aces",
    "MIN": "Minnesota Lynx",
    "NYL": "New York Liberty",
    "PDX": "Portland Fire",
    "PHX": "Phoenix Mercury",
    "SEA": "Seattle Storm",
    "TOR": "Toronto Tempo",
    "WAS": "Washington Mystics",
}
FULL_TO_ABB = {v: k for k, v in ABB_TO_FULL.items()}

# ESPN sometimes uses slightly different name spellings
ESPN_NAME_MAP = {
    "Atlanta Dream":              "Atlanta Dream",
    "Chicago Sky":                "Chicago Sky",
    "Connecticut Sun":            "Connecticut Sun",
    "Dallas Wings":               "Dallas Wings",
    "Indiana Fever":              "Indiana Fever",
    "Los Angeles Sparks":         "Los Angeles Sparks",
    "Minnesota Lynx":             "Minnesota Lynx",
    "New York Liberty":           "New York Liberty",
    "Phoenix Mercury":            "Phoenix Mercury",
    "Seattle Storm":              "Seattle Storm",
    "Washington Mystics":         "Washington Mystics",
    "Las Vegas Aces":             "Las Vegas Aces",
    "Golden State Valkyries":     "Golden State Valkyries",
    "Portland Fire":              "Portland Fire",
    "Toronto Tempo":              "Toronto Tempo",
}

# ── Arena factors ────────────────────────────────────────────────────────────────
# Pace/scoring environment factor per arena (1.0 = league average).
# WNBA is less well-documented than MLB park factors — these are reasonable estimates
# based on arena size, altitude, and historical scoring patterns.
ARENA_FACTOR = {
    "ATL": 1.02,  # State Farm Arena — large, good atmosphere
    "CHI": 0.99,  # Wintrust Arena — enclosed, slightly defensive
    "CON": 1.00,  # Mohegan Sun Arena
    "DAL": 1.01,  # College Park Center
    "IND": 1.02,  # Gainbridge Fieldhouse — Fever home crowds surged with Caitlin Clark
    "LA":  0.98,  # Crypto.com Arena — large, can feel empty
    "MIN": 1.01,  # Target Center
    "NYL": 1.03,  # Barclays Center — sold out regularly, electric atmosphere
    "PHX": 1.00,  # Footprint Center
    "SEA": 1.02,  # Climate Pledge Arena — rowdy home crowds
    "WAS": 0.99,  # Capital One Arena
    "LVA": 1.03,  # Michelob Ultra Arena — top home environment
    "GS":  1.01,  # Chase Center
    "PDX": 1.00,  # Moda Center
    "TOR": 1.00,  # Scotiabank Arena
}

def arena_factor(home_abb: str) -> float:
    return ARENA_FACTOR.get(home_abb, 1.00)


# ── Data fetching via ESPN public API ───────────────────────────────────────────
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"

@st.cache_resource(show_spinner=False)
def http_session() -> requests.Session:
    """One pooled HTTP session for the whole app (connection keep-alive across
    all ESPN calls, one automatic retry on connection errors)."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16,
                                            max_retries=1)
    s.mount("https://", adapter)
    return s


def _month_chunks(start: datetime.datetime,
                  end: datetime.datetime) -> list[tuple]:
    """Split [start, end] into calendar-month-aligned (chunk_start, chunk_end)
    pairs, inclusive on both ends."""
    chunks = []
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        chunk_end  = min(next_month - datetime.timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + datetime.timedelta(days=1)
    return chunks


def _final_games_from_events(events: list) -> dict:
    """Extract completed games from raw ESPN scoreboard events.
    Returns {event_id: {date, h_abb, a_abb, h, a}} (dict for de-duplication)."""
    games = {}
    for ev in events:
        try:
            if ev.get("status", {}).get("type", {}).get("name", "") != "STATUS_FINAL":
                continue
            comp  = ev.get("competitions", [{}])[0]
            comps = comp.get("competitors", [])
            if len(comps) < 2:
                continue
            home    = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
            away    = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
            h_abb   = ESPN_ABB_TO_ABB.get(home.get("team", {}).get("abbreviation", ""), "")
            a_abb   = ESPN_ABB_TO_ABB.get(away.get("team", {}).get("abbreviation", ""), "")
            h_score = int(home.get("score", 0) or 0)
            a_score = int(away.get("score", 0) or 0)
            if (h_score == 0 and a_score == 0) or not h_abb or not a_abb:
                continue
            key = ev.get("id") or f"{h_abb}-{a_abb}-{ev.get('date', '')}"
            games[key] = {
                # UTC date from the event; only used to order games, where a
                # late-tip rolling into the next UTC day is immaterial.
                "date":  ev.get("date", "")[:10],
                "h_abb": h_abb, "a_abb": a_abb,
                "h":     h_score, "a":  a_score,
            }
        except Exception:
            continue
    return games


def _scoreboard_events_for_day(sess: requests.Session, date_str: str) -> list:
    try:
        resp = sess.get(f"{ESPN_BASE}/scoreboard",
                        params={"dates": date_str}, timeout=8)
        return resp.json().get("events", [])
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_completed_games(season: int) -> list[dict]:
    """
    Every completed regular-season game for the season, sorted oldest → newest:
    [{date, h_abb, a_abb, h, a}, ...]

    This is the single scoreboard scan that opponent stats, standings, and
    recent form are all derived from. It uses ESPN's date-range query
    (dates=YYYYMMDD-YYYYMMDD&limit=N — same endpoint, verified to return every
    event in the span), so a full season costs 1 request per month (≤6 total)
    instead of the old 1 request per day (~150, and the old code ran that scan
    separately for opp stats AND recent form). If a range request errors or
    comes back empty, that month falls back to concurrent per-day requests so
    a quirk in the range API can never silently lose data.
    Falls back to last season if the selected season has no completed games yet.
    """
    sess = http_session()
    for yr in (season, season - 1):
        # WNBA regular season starts May 8 (skips early-May preseason games).
        start = datetime.datetime(yr, 5, 8)
        end   = min(datetime.datetime(yr, 10, 1), datetime.datetime.today())
        if start > end:
            continue

        games: dict = {}
        for c_start, c_end in _month_chunks(start, end):
            events = None
            try:
                resp = sess.get(
                    f"{ESPN_BASE}/scoreboard",
                    params={"dates": f"{c_start:%Y%m%d}-{c_end:%Y%m%d}",
                            "limit": 400},
                    timeout=10)
                events = resp.json().get("events", [])
            except Exception:
                events = None

            if not events:
                # Range query failed for this month — fetch its days in
                # parallel so the fallback path stays fast too.
                days = [(c_start + datetime.timedelta(days=i)).strftime("%Y%m%d")
                        for i in range((c_end - c_start).days + 1)]
                events = []
                with ThreadPoolExecutor(max_workers=12) as pool:
                    futures = [pool.submit(_scoreboard_events_for_day, sess, d)
                               for d in days]
                    for fut in as_completed(futures):
                        events.extend(fut.result())

            games.update(_final_games_from_events(events))

        if games:
            return sorted(games.values(), key=lambda g: g["date"])
    return []


def fetch_opp_stats(season: int) -> dict:
    """
    Per-team opp_pts_pg, margin_pg, W, L, W_PCT — derived from the shared game
    list with zero extra HTTP. (Name kept from the old implementation, which
    ran its own full per-day scoreboard scan to compute the same thing.)
    """
    own_map, opp_map, wins_map, loss_map = {}, {}, {}, {}
    for g in fetch_completed_games(season):
        own_map.setdefault(g["h_abb"], []).append(g["h"])
        opp_map.setdefault(g["h_abb"], []).append(g["a"])
        own_map.setdefault(g["a_abb"], []).append(g["a"])
        opp_map.setdefault(g["a_abb"], []).append(g["h"])
        if g["h"] > g["a"]:
            wins_map[g["h_abb"]] = wins_map.get(g["h_abb"], 0) + 1
            loss_map[g["a_abb"]] = loss_map.get(g["a_abb"], 0) + 1
        else:
            wins_map[g["a_abb"]] = wins_map.get(g["a_abb"], 0) + 1
            loss_map[g["h_abb"]] = loss_map.get(g["h_abb"], 0) + 1

    result = {}
    for abb in own_map:
        opps = opp_map.get(abb, [])
        owns = own_map.get(abb, [])
        w    = wins_map.get(abb, 0)
        l    = loss_map.get(abb, 0)
        gp   = w + l
        result[abb] = {
            "opp_pts_pg": round(sum(opps) / len(opps), 1) if opps else 0.0,
            "margin_pg":  round((sum(owns) / len(owns)) - (sum(opps) / len(opps)), 2) if opps else 0.0,
            "W":     w,
            "L":     l,
            "W_PCT": w / gp if gp > 0 else 0.5,
        }
    return result


def fetch_standings(season: int) -> dict:
    """W-L records keyed by full team name, derived from the shared game list."""
    return {
        ABB_TO_FULL.get(abb, abb): {"W": v["W"], "L": v["L"], "W_PCT": v["W_PCT"]}
        for abb, v in fetch_opp_stats(season).items()
    }


def fetch_recent_stats(season: int, last_n: int = 5) -> dict:
    """
    Recent-form averages over each team's last N completed games, derived from
    the shared game list. Because this is now a pure in-memory pass, moving the
    form-window slider is free — the old version was cache-keyed on N and
    re-ran the entire per-day scoreboard scan every time the slider changed.
    (Box stats like FG% aren't in scoreboard data, so recent form covers
    pts/opp_pts/margin/record and season averages fill in the rest — same
    approximation as before.)
    """
    games_by_team = {}
    for g in fetch_completed_games(season):          # oldest → newest
        games_by_team.setdefault(g["h_abb"], []).append((g["h"], g["a"]))
        games_by_team.setdefault(g["a_abb"], []).append((g["a"], g["h"]))

    result = {}
    for abb, scores in games_by_team.items():
        recent = scores[-last_n:]                    # newest N games
        n = len(recent)
        if n < 1:
            continue
        pts_pg = round(sum(s[0] for s in recent) / n, 1)
        opp_pg = round(sum(s[1] for s in recent) / n, 1)
        w      = sum(1 for s in recent if s[0] > s[1])
        result[abb] = {
            "G":          n,
            "pts_pg":     pts_pg,
            "opp_pts_pg": opp_pg,
            "margin_pg":  round(pts_pg - opp_pg, 2),
            "W":          w,
            "L":          n - w,
            "W_PCT":      w / n if n > 0 else 0.5,
        }
    return result


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_todays_games() -> tuple[list, str]:
    """Fetch today's WNBA games from ESPN."""
    today = datetime.datetime.today().strftime("%Y%m%d")
    try:
        resp = http_session().get(f"{ESPN_BASE}/scoreboard",
                                  params={"dates": today}, timeout=10)
        return resp.json().get("events", []), ""
    except Exception as e:
        return [], str(e)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_stats(season: int) -> dict:
    """
    Fetch season team statistics from ESPN for all WNBA teams.
    Returns {abb: {pts_pg, opp_pts_pg, fg_pct, fg3_pct, ft_pct, reb_pg, ast_pg,
    tov_pg, ...}}.

    Teams are fetched concurrently: each team may need up to 4 attempts to find
    the season/seasontype combination ESPN actually populated, so doing all
    teams in parallel turns a worst case of ~60 sequential requests into the
    wall time of the slowest single team.
    """
    sess = http_session()
    try:
        resp  = sess.get(f"{ESPN_BASE}/teams", timeout=10)
        teams = resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    except Exception:
        return {}

    def fetch_one(entry: dict):
        team = entry.get("team", {})
        tid  = team.get("id", "")
        # Resolve abbreviation: try ESPN ID first, then ESPN raw abbr map, then name
        raw_abb = team.get("abbreviation", "")
        abb = (ESPN_ID_TO_ABB.get(tid)
               or ESPN_ABB_TO_ABB.get(raw_abb)
               or FULL_TO_ABB.get(ESPN_NAME_MAP.get(team.get("displayName", ""),
                                                    team.get("displayName", ""))))
        if not abb:
            return None, None

        try:
            stats_url = f"{ESPN_BASE}/teams/{tid}/statistics"
            # Try season+1 first (ESPN may label current season by next year),
            # then current season, then seasontype 3
            raw = {}
            for season_try, stype in [(season + 1, 2), (season, 2),
                                      (season + 1, 3), (season, 3)]:
                sr = sess.get(stats_url,
                              params={"season": season_try, "seasontype": stype},
                              timeout=10)
                if sr.status_code == 200 and sr.text:
                    candidate = sr.json()
                    cats = candidate.get("results", {}).get("stats", {}).get("categories", [])
                    # Check if we actually got stats with real values
                    has_data = any(
                        s.get("value", 0) != 0
                        for cat in cats for s in cat.get("stats", [])
                        if s.get("name") == "gamesPlayed"
                    )
                    if has_data:
                        raw = candidate
                        break

            # ESPN stats live at: results -> stats -> categories (NOT splits -> categories)
            categories = raw.get("results", {}).get("stats", {}).get("categories", [])
            stat_map = {}
            for cat in categories:
                for stat in cat.get("stats", []):
                    abbr = stat.get("abbreviation", "")
                    name = stat.get("name", "")
                    val  = stat.get("value", 0.0)
                    if abbr: stat_map[abbr] = val
                    if name: stat_map[name] = val

            def g(*keys):
                for k in keys:
                    v = stat_map.get(k)
                    if v is not None:
                        try:
                            fv = float(v)
                            if fv != 0: return fv
                        except (TypeError, ValueError):
                            pass
                return 0.0

            # Values are already per-game averages.
            # Percentages are on 0-100 scale (e.g. FG% = 45.6) — divide by 100.
            pts     = g("avgPoints",    "PTS")
            fg_pct  = g("FG%",          "fieldGoalPct")               / 100.0
            fg3_pct = g("3P%",          "threePointPct",
                        "threePointFieldGoalPct")                      / 100.0
            ft_pct  = g("FT%",          "freeThrowPct")               / 100.0
            reb     = g("avgRebounds",  "REB")
            ast     = g("avgAssists",   "AST")
            tov     = g("avgTurnovers", "TO")
            stl     = g("avgSteals",    "STL")
            blk     = g("avgBlocks",    "BLK")
            off_reb = g("avgOffensiveRebounds", "OR")
            def_reb = g("avgDefensiveRebounds", "DR")
            gp      = max(int(g("gamesPlayed",  "GP")), 1)

            # eFG% = FG% + 0.5 * (3PM / FGA)
            fgm3    = g("avgThreePointFieldGoalsMade", "3PM")
            fga     = g("avgFieldGoalsAttempted",      "FGA")
            efg_pct = (fg_pct + 0.5 * (fgm3 / fga)) if fga else fg_pct

            # Opponent pts not in this endpoint — left as 0, filled from
            # the shared game scan in the sidebar merge step
            return abb, {
                "G":         gp,
                "pts_pg":    round(pts,     1),
                "opp_pts_pg":0.0,
                "margin_pg": 0.0,
                "fg_pct":    round(fg_pct,  3),
                "fg3_pct":   round(fg3_pct, 3),
                "ft_pct":    round(ft_pct,  3),
                "efg_pct":   round(efg_pct, 3),
                "reb_pg":    round(reb,     1),
                "ast_pg":    round(ast,     1),
                "tov_pg":    round(tov,     1),
                "stl_pg":    round(stl,     1),
                "blk_pg":    round(blk,     1),
                "off_reb_pg":round(off_reb, 1),
                "def_reb_pg":round(def_reb, 1),
            }
        except Exception:
            return None, None

    all_stats = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, entry) for entry in teams]
        for fut in as_completed(futures):
            abb, stats = fut.result()
            if abb and stats:
                all_stats[abb] = stats
    return all_stats



def parse_espn_game(event: dict) -> dict | None:
    """
    Parse an ESPN event dict into a normalized game dict.
    Returns None if parsing fails.
    """
    try:
        comp  = event.get("competitions", [{}])[0]
        comps = comp.get("competitors", [])
        if len(comps) < 2:
            return None

        home_comp = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
        away_comp = next((c for c in comps if c.get("homeAway") == "away"), comps[1])

        def team_abb(comp_dict):
            t   = comp_dict.get("team", {})
            raw = t.get("abbreviation", "")
            return ESPN_ABB_TO_ABB.get(raw, raw)

        home_abb  = team_abb(home_comp)
        away_abb  = team_abb(away_comp)
        home_name = ABB_TO_FULL.get(home_abb, home_comp.get("team", {}).get("displayName", home_abb))
        away_name = ABB_TO_FULL.get(away_abb, away_comp.get("team", {}).get("displayName", away_abb))

        home_score = int(home_comp.get("score", 0) or 0)
        away_score = int(away_comp.get("score", 0) or 0)

        status_type = event.get("status", {}).get("type", {})
        status      = status_type.get("name", "")
        date_str    = event.get("date", "")[:10]  # YYYY-MM-DD
        venue_name  = comp.get("venue", {}).get("fullName", "")

        return {
            "home_abb":    home_abb,
            "away_abb":    away_abb,
            "home_name":   home_name,
            "away_name":   away_name,
            "home_score":  home_score,
            "away_score":  away_score,
            "status":      status,
            "date":        date_str,
            "venue":       venue_name,
            "game_id":     event.get("id", ""),
        }
    except Exception:
        return None


# ── Model helpers ───────────────────────────────────────────────────────────────
def sf(d: dict, *keys) -> float:
    for k in keys:
        try:
            v = float(d.get(k, 0) or 0)
            if v:
                return v
        except (TypeError, ValueError):
            pass
    return 0.0

# WNBA league-average baselines (2023–2024 approximate)
LEAGUE_AVG = {
    "pts_pg":     82.0,
    "opp_pts_pg": 82.0,
    "margin_pg":  0.0,
    "fg_pct":     0.430,
    "fg3_pct":    0.335,
    "ft_pct":     0.790,
    "efg_pct":    0.475,
    "reb_pg":     33.5,
    "ast_pg":     19.5,
    "tov_pg":     14.5,
    "stl_pg":     7.0,
    "blk_pg":     3.5,
    "off_reb_pg": 10.0,
    "def_reb_pg": 23.5,
}

LEAGUE_STD = {
    "pts_pg":     6.0,
    "opp_pts_pg": 5.5,
    "margin_pg":  6.0,
    "fg_pct":     0.020,
    "fg3_pct":    0.025,
    "ft_pct":     0.030,
    "efg_pct":    0.025,
    "reb_pg":     2.5,
    "ast_pg":     2.5,
    "tov_pg":     1.5,
    "stl_pg":     1.0,
    "blk_pg":     0.8,
    "off_reb_pg": 1.5,
    "def_reb_pg": 2.0,
}

def norm_vs_league(val: float, avg: float, std: float, inv: bool = False) -> float:
    """Normalise a stat vs league average. Returns 0–1, 0.5 = league avg."""
    z = (val - avg) / std if std else 0
    z = max(-3, min(3, z))
    score = (z + 3) / 6
    return round(1 - score if inv else score, 3)


def score_team(abb: str, team_stats: dict, standings: dict) -> dict:
    s         = team_stats.get(abb, {})
    full_name = ABB_TO_FULL.get(abb, abb)
    rec       = standings.get(full_name, {})
    w         = rec.get("W", 0)
    l         = rec.get("L", 0)
    wpct      = rec.get("W_PCT", 0.5)
    gp        = max(float(s.get("G", 1) or 1), 1)
    return {
        "name":       full_name,
        "abb":        abb,
        "G":          gp,
        "pts_pg":     sf(s, "pts_pg"),
        "opp_pts_pg": sf(s, "opp_pts_pg"),
        "margin_pg":  sf(s, "margin_pg"),
        "fg_pct":     sf(s, "fg_pct"),
        "fg3_pct":    sf(s, "fg3_pct"),
        "ft_pct":     sf(s, "ft_pct"),
        "efg_pct":    sf(s, "efg_pct"),
        "reb_pg":     sf(s, "reb_pg"),
        "ast_pg":     sf(s, "ast_pg"),
        "tov_pg":     sf(s, "tov_pg"),
        "stl_pg":     sf(s, "stl_pg"),
        "blk_pg":     sf(s, "blk_pg"),
        "off_reb_pg": sf(s, "off_reb_pg"),
        "def_reb_pg": sf(s, "def_reb_pg"),
        "w": w, "l": l, "wpct": round(wpct * 100, 1),
    }




def blend_stats(season_s: dict, recent_s: dict, w_season: float, w_recent: float) -> dict:
    """Blend season-long and recent-form stats using given weights."""
    if not recent_s or recent_s.get("G", 0) < 1:
        return season_s
    blended = dict(season_s)
    for key in ("pts_pg", "opp_pts_pg", "margin_pg"):
        sv = float(season_s.get(key, 0) or 0)
        rv = float(recent_s.get(key, 0) or 0)
        if rv != 0:
            blended[key] = round(sv * w_season + rv * w_recent, 2)
    # For shooting/rebounding/etc, we only have season data — keep as-is
    return blended

def score_vs_league(s: dict) -> dict:
    def n(key, inv=False):
        return norm_vs_league(s.get(key, LEAGUE_AVG[key]),
                              LEAGUE_AVG[key], LEAGUE_STD[key], inv)
    return {
        # Offense (higher is better)
        "pts_score":    n("pts_pg"),
        "efg_score":    n("efg_pct"),
        "fg3_score":    n("fg3_pct"),
        "ft_score":     n("ft_pct"),
        "ast_score":    n("ast_pg"),
        "oreb_score":   n("off_reb_pg"),
        # Defense (lower opponent pts / turnovers = better)
        "def_score":    n("opp_pts_pg", inv=True),
        "tov_score":    n("tov_pg",     inv=True),   # fewer turnovers = good
        "dreb_score":   n("def_reb_pg"),
        "stl_score":    n("stl_pg"),
        "blk_score":    n("blk_pg"),
        # Overall
        "margin_score": n("margin_pg"),
        # Win% z-scored like every other component so all composite inputs
        # share one scale. League-wide W% spread is ~0.165 std across teams;
        # with the (z+3)/6 normalisation that lands within ~0.01 of the old
        # raw-wpct value for typical records, so outputs barely move — the
        # gain is that the scale assumption is now explicit instead of a
        # numeric coincidence.
        "wpct_score":   norm_vs_league(s.get("wpct", 50) / 100, 0.5, 0.165),
    }


def build_composite(sa: dict, sb: dict):
    """Return (off_a, def_a, rec_a, off_b, def_b, rec_b) for two teams."""
    la = score_vs_league(sa)
    lb = score_vs_league(sb)

    # Offense: scoring (35%), EFG (25%), 3P% (15%), FT% (10%), assists (10%), OREB (5%)
    off_a = (la["pts_score"] * 0.35 + la["efg_score"]  * 0.25 +
             la["fg3_score"] * 0.15 + la["ft_score"]   * 0.10 +
             la["ast_score"] * 0.10 + la["oreb_score"] * 0.05)
    off_b = (lb["pts_score"] * 0.35 + lb["efg_score"]  * 0.25 +
             lb["fg3_score"] * 0.15 + lb["ft_score"]   * 0.10 +
             lb["ast_score"] * 0.10 + lb["oreb_score"] * 0.05)

    # Defense: opp pts (40%), DREB (25%), turnovers (20%), steals (10%), blocks (5%)
    def_a = (la["def_score"]  * 0.40 + la["dreb_score"] * 0.25 +
             la["tov_score"]  * 0.20 + la["stl_score"]  * 0.10 +
             la["blk_score"]  * 0.05)
    def_b = (lb["def_score"]  * 0.40 + lb["dreb_score"] * 0.25 +
             lb["tov_score"]  * 0.20 + lb["stl_score"]  * 0.10 +
             lb["blk_score"]  * 0.05)

    # Record: W% (60%) + scoring margin (40%)
    rec_a = la["wpct_score"] * 0.60 + la["margin_score"] * 0.40
    rec_b = lb["wpct_score"] * 0.60 + lb["margin_score"] * 0.40

    return off_a, def_a, rec_a, off_b, def_b, rec_b


# Steepness of the logistic that maps composite-score edge → win probability.
# Composite scores live on a 0–1 scale (0.5 = league average), so the edge
# d = sc_a - sc_b is typically within ±0.10 for ordinary matchups and ~±0.30
# for extreme best-vs-worst mismatches. With K = 6.0:
#     d = 0.05 → 57%      d = 0.10 → 65%      d = 0.30 → 86%
# which matches the realistic WNBA range (even elite teams rarely price above
# ~85–90% against the league's worst). The old score-ratio formula
# (sc_a / (sc_a + sc_b)) compressed everything into ~45–55%: with both scores
# hovering near 0.5, a 0.60-vs-0.50 blowout matchup came out as only 54.5%,
# while HOME_BOOST moved the needle a token ~1.7pp. K is the single
# calibration knob if probabilities ever run systematically hot or cold.
LOGISTIC_K = 6.0

def calc_prob(sa, sb, home, w_off, w_def, w_rec,
              ra=None, rb=None, w_season=1.0, w_recent=0.0):
    ea = blend_stats(sa, ra, w_season, w_recent) if ra else sa
    eb = blend_stats(sb, rb, w_season, w_recent) if rb else sb
    off_a, def_a, rec_a, off_b, def_b, rec_b = build_composite(ea, eb)
    sc_a = off_a * w_off + def_a * w_def + rec_a * w_rec
    sc_b = off_b * w_off + def_b * w_def + rec_b * w_rec
    if home == "home":   sc_a += HOME_BOOST
    elif home == "away": sc_b += HOME_BOOST
    # Logistic mapping of the score edge — see the LOGISTIC_K note above.
    p_a = 1.0 / (1.0 + math.exp(-LOGISTIC_K * (sc_a - sc_b)))
    return p_a, 1.0 - p_a


def calc_spread(sa, sb, home, w_off, w_def, w_rec, af: float = 1.0,
               ra=None, rb=None, w_season=1.0, w_recent=0.0):
    """Project point totals and spread."""
    ea = blend_stats(sa, ra, w_season, w_recent) if ra else sa
    eb = blend_stats(sb, rb, w_season, w_recent) if rb else sb
    _, def_a, _, _, def_b, _ = build_composite(ea, eb)

    supp    = 4.0   # points suppressed by a 1-std defense advantage
    proj_h  = ea["pts_pg"] - (def_b - 0.5) * supp * 2
    proj_a  = eb["pts_pg"] - (def_a - 0.5) * supp * 2

    if home == "home":
        proj_h += HOME_PTS

    # Apply arena pace/scoring factor
    proj_h = proj_h * af
    proj_a = proj_a * af

    proj_h  = max(55.0, proj_h)
    proj_a  = max(55.0, proj_a)
    margin  = proj_h - proj_a
    winner  = sa["name"] if margin >= 0 else sb["name"]

    def snap(v): return round(round(v * 2) / 2, 1)
    proj_h  = snap(proj_h)
    proj_a  = snap(proj_a)
    total   = snap(proj_h + proj_a)

    return proj_h, proj_a, winner, margin, total


def calc_confidence(sa, sb, pct_h, pct_a, margin_winner, prob_winner):
    models_agree  = prob_winner == margin_winner
    prob_gap      = abs(pct_h - pct_a)
    # Thresholds retuned for the logistic win-prob scale: probabilities now
    # span a realistic ~15–86% instead of the old ratio model's compressed
    # ~45–55% band, so the old 12pp/5pp cutoffs would flag nearly every game
    # as "strong". 24pp = a 62/38 game, 10pp = 55/45 — roughly the same
    # selectivity the old cutoffs had on the old scale.
    prob_strength = "strong" if prob_gap >= 24 else ("moderate" if prob_gap >= 10 else "narrow")

    off_leader = sa["name"] if sa["pts_pg"]  > sb["pts_pg"]  else sb["name"]
    rec_leader = sa["name"] if sa["wpct"]    > sb["wpct"]    else sb["name"]
    split      = off_leader != rec_leader

    if models_agree and prob_strength == "strong" and not split:
        level, emoji, color = "High",       "🟢", "#f60"
    elif models_agree and prob_strength in ("strong", "moderate"):
        level, emoji, color = "Moderate",   "🟡", "#f5c842"
    elif models_agree:
        level, emoji, color = "Low",        "🟠", "#f5a623"
    else:
        level, emoji, color = "Conflicted", "🔴", "#ff5252"

    reasons = []
    if not models_agree:
        reasons.append(f"Win probability favors **{prob_winner}**, but the projected spread "
                       f"favors **{margin_winner}** — the models disagree.")
    if split:
        reasons.append(f"**{off_leader}** scores more per game but **{rec_leader}** has the "
                       f"better record — schedule difficulty may be a factor.")
    if prob_strength == "narrow" and models_agree:
        reasons.append(f"Both models agree on **{prob_winner}** but the edge is slim "
                       f"({prob_gap:.1f}pp) — a single hot-shooting quarter could flip this.")
    if not reasons:
        reasons.append(f"Both models consistently favor **{prob_winner}** with a "
                       f"{prob_strength} edge across offense, defense, and record.")
    return level, emoji, color, reasons


def arena_factor_reason(venue: str, af: float) -> str | None:
    if af >= 1.025:
        return (f"**{venue}** is a high-scoring environment (factor: {af:.2f}x) "
                f"— expect elevated totals and potentially wider margins.")
    if af <= 0.985:
        return (f"**{venue}** tends to suppress scoring (factor: {af:.2f}x) "
                f"— games here often play slower and stay closer.")
    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 WNBA Daily Slate")

    CURRENT_YEAR = datetime.datetime.now().year
    SEASON = st.selectbox("Season", [CURRENT_YEAR, CURRENT_YEAR - 1], index=0)
    st.caption(f"Season: {SEASON} · ESPN · refreshes every 30 min")

    st.markdown("---")
    st.markdown("##### Settings")
    home_display = st.checkbox("Show home-court advantage", value=True)

    st.markdown("---")
    st.markdown("##### Recent form")
    ng = st.select_slider("Form window (games)", options=[3, 5, 7, 10], value=5)
    w_season_pct = st.slider("Season weight",      0, 100, 60, step=5)
    w_recent_pct = st.slider("Recent form weight", 0, 100, 40, step=5)
    form_total   = (w_season_pct + w_recent_pct) or 1
    w_season = w_season_pct / form_total
    w_recent = w_recent_pct / form_total

    st.markdown("---")
    st.markdown("##### Model weights")
    w_off_raw = st.slider("Offense weight",  0, 100, 40, step=5)
    w_def_raw = st.slider("Defense weight",  0, 100, 35, step=5)
    w_rec_raw = st.slider("Record weight",   0, 100, 25, step=5)
    total_w   = (w_off_raw + w_def_raw + w_rec_raw) or 1
    w_off = w_off_raw / total_w
    w_def = w_def_raw / total_w
    w_rec = w_rec_raw / total_w

    st.markdown("---")
    with st.spinner("Loading WNBA stats..."):
        team_stats_data = fetch_team_stats(SEASON)
        standings_data  = fetch_standings(SEASON)
        opp_stats_data  = fetch_opp_stats(SEASON)
        recent_stats_data = fetch_recent_stats(SEASON, last_n=ng)
        # Merge opp_pts_pg and margin_pg into team_stats_data
        for abb, opp in opp_stats_data.items():
            if abb in team_stats_data:
                team_stats_data[abb]["opp_pts_pg"] = opp["opp_pts_pg"]
                team_stats_data[abb]["margin_pg"]  = opp["margin_pg"]

    if team_stats_data:
        st.success(f"✅ Stats loaded for {len(team_stats_data)} teams")
    else:
        st.error("⚠️ Could not load team stats")

    if standings_data:
        st.caption(f"📋 Standings loaded for {len(standings_data)} teams")
    else:
        st.caption("📋 Standings not yet available (normal early in season)")

    # Archive today's stats (see save_snapshot's note on why the archive stays).
    if team_stats_data:
        save_snapshot(team_stats_data, standings_data or {})


# ═══════════════════════════════════════════════════════════════════════════════
# Today's Games
# ═══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.header("Today's WNBA Slate")

    if not team_stats_data:
        st.error("Unable to load team stats. ESPN API may be unavailable.")
        st.stop()

    games_raw, err = fetch_todays_games()
    games = [g for g in [parse_espn_game(ev) for ev in games_raw] if g]

    if err:
        st.warning(f"API error: {err}")

    if not games:
        st.info("No WNBA games scheduled today — check back on a game day! 🏀")
    else:
        st.caption(f"{len(games)} game(s) today · {datetime.datetime.today().strftime('%A, %B %d, %Y')}")

        for game in games:
            home_abb = game["home_abb"]
            away_abb = game["away_abb"]
            af       = arena_factor(home_abb)

            sa = score_team(home_abb, team_stats_data, standings_data)
            sb = score_team(away_abb, team_stats_data, standings_data)
            ra = recent_stats_data.get(home_abb)
            rb = recent_stats_data.get(away_abb)

            # score_team floors G at 1, so the old `G < 1` guard could never
            # fire — a team missing from the stats dict silently produced
            # all-zero inputs. Check dict membership instead.
            if home_abb not in team_stats_data or away_abb not in team_stats_data:
                st.warning(f"⚠️ No season stats yet for {game['home_name']} or {game['away_name']}")
                continue

            pct_h, pct_a = calc_prob(sa, sb, "home", w_off, w_def, w_rec,
                                     ra, rb, w_season, w_recent)
            pct_h_i = round(pct_h * 100)
            pct_a_i = round(pct_a * 100)

            proj_h, proj_a, margin_winner, margin, total_pts = calc_spread(
                sa, sb, "home", w_off, w_def, w_rec, af,
                ra, rb, w_season, w_recent)

            prob_winner  = sa["name"] if pct_h >= pct_a else sb["name"]
            conf_l, conf_e, conf_c, reasons = calc_confidence(
                sa, sb, pct_h_i, pct_a_i, margin_winner, prob_winner)

            af_reason = arena_factor_reason(game["venue"], af)
            if af_reason:
                reasons.append(af_reason)

            spread_val = round(round(abs(margin) * 2) / 2, 1)
            fav_spread = margin_winner
            dog_spread = sb["name"] if margin_winner == sa["name"] else sa["name"]

            status = game.get("status", "")
            is_live   = status == "STATUS_IN_PROGRESS"
            is_final  = status == "STATUS_FINAL"

            with st.container():
                st.markdown("---")
                # Header row
                hc1, hc2, hc3 = st.columns([3, 1, 1])
                with hc1:
                    home_label = f"🏠 {game['home_name']}" if home_display else game['home_name']
                    st.subheader(f"{game['away_name']} @ {home_label}")
                    if game.get("venue"):
                        st.caption(f"📍 {game['venue']}  ·  Arena factor: {af:.2f}x")
                with hc2:
                    if is_live:
                        st.markdown("🔴 **LIVE**")
                    elif is_final:
                        st.markdown(f"**Final: {game['away_score']} – {game['home_score']}**")
                with hc3:
                    st.markdown(f"<span style='color:{conf_c};font-weight:700;font-size:1.1rem'>"
                                f"{conf_e} {conf_l}</span>", unsafe_allow_html=True)

                # Win probability bars
                col_a, col_b = st.columns(2)
                with col_a:
                    bar_color = "#f60" if pct_h_i >= pct_a_i else "#555"
                    st.markdown(
                        f"<div style='background:{bar_color};border-radius:8px;padding:10px 14px;"
                        f"text-align:center'>"
                        f"<div style='font-size:0.75rem;color:#fff;opacity:0.8'>HOME</div>"
                        f"<div style='font-size:2.2rem;font-weight:800;color:#fff'>{pct_h_i}%</div>"
                        f"<div style='font-size:0.9rem;color:#fff'>{sa['name']}</div></div>",
                        unsafe_allow_html=True)
                with col_b:
                    bar_color = "#f60" if pct_a_i > pct_h_i else "#555"
                    st.markdown(
                        f"<div style='background:{bar_color};border-radius:8px;padding:10px 14px;"
                        f"text-align:center'>"
                        f"<div style='font-size:0.75rem;color:#fff;opacity:0.8'>AWAY</div>"
                        f"<div style='font-size:2.2rem;font-weight:800;color:#fff'>{pct_a_i}%</div>"
                        f"<div style='font-size:0.9rem;color:#fff'>{sb['name']}</div></div>",
                        unsafe_allow_html=True)

                st.markdown("")

                # Spread and total
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Projected score", f"{proj_a:.0f} – {proj_h:.0f}")
                m2.metric("Projected total", f"{total_pts:.0f} pts")
                m3.metric("Spread pick",
                          f"{fav_spread} -{spread_val}",
                          f"+{spread_val} {dog_spread}")
                m4.metric("Win prob pick",
                          prob_winner,
                          f"Edge: {abs(pct_h_i - pct_a_i)}pp")

                # Stats comparison
                st.markdown('<p class="section-head">Season stats comparison</p>',
                            unsafe_allow_html=True)

                def better(a_val, b_val, inv=False):
                    if inv:
                        a_better = a_val < b_val
                    else:
                        a_better = a_val > b_val
                    a_cls = "stat-better" if a_better else "stat-worse"
                    b_cls = "stat-better" if not a_better else "stat-worse"
                    return a_cls, b_cls

                stat_rows = [
                    ("Pts/g",    "pts_pg",     False, "{:.1f}"),
                    ("Opp Pts/g","opp_pts_pg", True,  "{:.1f}"),
                    ("Margin/g", "margin_pg",  False, "{:+.1f}"),
                    ("FG%",      "fg_pct",     False, "{:.1%}"),
                    ("3P%",      "fg3_pct",    False, "{:.1%}"),
                    ("FT%",      "ft_pct",     False, "{:.1%}"),
                    ("Reb/g",    "reb_pg",     False, "{:.1f}"),
                    ("Ast/g",    "ast_pg",     False, "{:.1f}"),
                    ("Tov/g",    "tov_pg",     True,  "{:.1f}"),
                    ("Stl/g",    "stl_pg",     False, "{:.1f}"),
                    ("Blk/g",    "blk_pg",     False, "{:.1f}"),
                    ("Record",   None,         False, ""),
                ]

                c1, c2, c3 = st.columns([2, 1.5, 1.5])
                c1.markdown(f"**Stat**")
                c2.markdown(f"**{sa['name'].split()[-1]}** (home)")
                c3.markdown(f"**{sb['name'].split()[-1]}** (away)")

                for label, key, inv, fmt in stat_rows:
                    if key is None:
                        av = f"{sa['w']}-{sa['l']} ({sa['wpct']:.1f}%)"
                        bv = f"{sb['w']}-{sb['l']} ({sb['wpct']:.1f}%)"
                        ac, bc = better(sa["wpct"], sb["wpct"])
                    else:
                        av_raw = sa.get(key, 0)
                        bv_raw = sb.get(key, 0)
                        av = fmt.format(av_raw) if fmt else str(av_raw)
                        bv = fmt.format(bv_raw) if fmt else str(bv_raw)
                        ac, bc = better(av_raw, bv_raw, inv)

                    r1, r2, r3 = st.columns([2, 1.5, 1.5])
                    r1.caption(label)
                    r2.markdown(f'<span class="{ac}">{av}</span>', unsafe_allow_html=True)
                    r3.markdown(f'<span class="{bc}">{bv}</span>', unsafe_allow_html=True)

                # Recent form callout
                if ra and rb:
                    st.markdown(f'<p class="section-head">Recent form (last {ng} games)</p>',
                                unsafe_allow_html=True)
                    rf1, rf2, rf3, rf4 = st.columns(4)
                    h_margin_delta = round((ra.get("margin_pg", 0) or 0) - (sa.get("margin_pg", 0) or 0), 1)
                    a_margin_delta = round((rb.get("margin_pg", 0) or 0) - (sb.get("margin_pg", 0) or 0), 1)
                    h_pts_delta    = round((ra.get("pts_pg", 0) or 0) - (sa.get("pts_pg", 0) or 0), 1)
                    a_pts_delta    = round((rb.get("pts_pg", 0) or 0) - (sb.get("pts_pg", 0) or 0), 1)
                    h_rec = f"{ra.get('W',0)}-{ra.get('L',0)}"
                    a_rec = f"{rb.get('W',0)}-{rb.get('L',0)}"
                    rf1.metric(
                        f"{sa['name'].split()[-1]} Pts/g (L{ng})",
                        f"{ra.get('pts_pg', 0):.1f}",
                        f"{h_pts_delta:+.1f} vs season",
                    )
                    rf2.metric(
                        f"{sa['name'].split()[-1]} Record (L{ng})",
                        h_rec,
                        f"Margin {ra.get('margin_pg', 0):+.1f} ({h_margin_delta:+.1f} vs season)",
                    )
                    rf3.metric(
                        f"{sb['name'].split()[-1]} Pts/g (L{ng})",
                        f"{rb.get('pts_pg', 0):.1f}",
                        f"{a_pts_delta:+.1f} vs season",
                    )
                    rf4.metric(
                        f"{sb['name'].split()[-1]} Record (L{ng})",
                        a_rec,
                        f"Margin {rb.get('margin_pg', 0):+.1f} ({a_margin_delta:+.1f} vs season)",
                    )

                # Confidence reasoning
                st.markdown('<p class="section-head">Confidence factors</p>',
                            unsafe_allow_html=True)
                for r in reasons:
                    st.markdown(f'<div class="confidence-reason">• {r}</div>',
                                unsafe_allow_html=True)

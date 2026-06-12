"""
WNBA Win Probability — Today's Slate
--------------------------------------
Install dependencies:
    pip install requests streamlit plotly numpy

Run:
    streamlit run wnba_app.py

Data source: ESPN public API (no API key required)
"""

import datetime
import json
import os
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Snapshot system ─────────────────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wnba_snapshots")

def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def snapshot_path(date_str: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"stats_{date_str}.json")

def save_snapshot(team_stats: dict, standings: dict, date_str: str = None):
    ensure_snapshot_dir()
    if date_str is None:
        date_str = datetime.datetime.today().strftime("%Y-%m-%d")
    path = snapshot_path(date_str)
    if os.path.exists(path):
        return  # Already saved for today
    payload = {
        "date":       date_str,
        "team_stats": team_stats,
        "standings":  standings,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

def load_snapshot(date_str: str) -> tuple[dict, dict]:
    path = snapshot_path(date_str)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("team_stats", {}), payload.get("standings", {})
    except Exception:
        return None, None

def get_best_snapshot_for_game(game_date_str: str) -> tuple[dict, dict, str]:
    ensure_snapshot_dir()
    try:
        game_dt = datetime.datetime.strptime(game_date_str, "%Y-%m-%d")
    except ValueError:
        return None, None, None

    snapshot_files = sorted([
        f for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("stats_") and f.endswith(".json")
    ], reverse=True)

    for fname in snapshot_files:
        snap_date_str = fname.replace("stats_", "").replace(".json", "")
        try:
            snap_dt = datetime.datetime.strptime(snap_date_str, "%Y-%m-%d")
            if snap_dt < game_dt:
                ts, st_d = load_snapshot(snap_date_str)
                if ts:
                    return ts, st_d, snap_date_str
        except ValueError:
            continue
    return None, None, None

def list_snapshots() -> list[str]:
    ensure_snapshot_dir()
    files = [
        f.replace("stats_", "").replace(".json", "")
        for f in os.listdir(SNAPSHOT_DIR)
        if f.startswith("stats_") and f.endswith(".json")
    ]
    return sorted(files)


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

SEASON     = datetime.datetime.now().year
HOME_BOOST = 0.035   # WNBA home advantage (slightly less than NBA)
HOME_PTS   = 2.5     # Expected home-court points bonus

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
    "16":     "SEA",
    "131935": "TOR",
    "16":     "WAS",   # conflict resolved below via abbr map
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

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_stats(season: int) -> dict:
    """
    Fetch season team statistics from ESPN for all WNBA teams.
    Returns {abb: {pts_pg, opp_pts_pg, fg_pct, fg3_pct, ft_pct, reb_pg, ast_pg, tov_pg, ...}}
    """
    all_stats = {}
    url = f"{ESPN_BASE}/teams"
    try:
        resp  = requests.get(url, timeout=10)
        teams = resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    except Exception:
        return all_stats

    for entry in teams:
        team = entry.get("team", {})
        tid  = team.get("id", "")
        # Resolve abbreviation: try ESPN ID first, then ESPN raw abbr map, then name
        raw_abb = team.get("abbreviation", "")
        abb = (ESPN_ID_TO_ABB.get(tid)
               or ESPN_ABB_TO_ABB.get(raw_abb)
               or FULL_TO_ABB.get(ESPN_NAME_MAP.get(team.get("displayName", ""), team.get("displayName", ""))))
        if not abb:
            continue

        try:
            stats_url = f"{ESPN_BASE}/teams/{tid}/statistics"
            # Try season+1 first (ESPN may label current season by next year),
            # then current season, then seasontype 3
            raw = {}
            for season_try, stype in [(season+1, 2), (season, 2), (season+1, 3), (season, 3)]:
                sr  = requests.get(stats_url,
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

            # Opponent pts not in this endpoint — left as 0, filled from scoreboard results
            opp_pts   = 0.0
            margin_pg = 0.0

            all_stats[abb] = {
                "G":         gp,
                "pts_pg":    round(pts,     1),
                "opp_pts_pg":round(opp_pts, 1),
                "margin_pg": margin_pg,
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
            continue

    return all_stats


def fetch_standings(season: int) -> dict:
    """
    Derives W-L from fetch_opp_stats (which scans the game log).
    Returns {full_name: {W, L, W_PCT}}.
    """
    opp = fetch_opp_stats(season)
    return {
        ABB_TO_FULL.get(abb, abb): {
            "W":     v["W"],
            "L":     v["L"],
            "W_PCT": v["W_PCT"],
        }
        for abb, v in opp.items()
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_opp_stats(season: int) -> dict:
    """
    Scan completed game results to compute opp_pts_pg, margin_pg, W, L for every team.
    Returns {abb: {opp_pts_pg, margin_pg, W, L, W_PCT}}.
    Also used by fetch_standings — both share this one scoreboard scan.
    """
    opp_map  = {}   # abb -> [opp scores]
    own_map  = {}   # abb -> [own scores]
    wins_map = {}
    loss_map = {}

    for yr in [season, season - 1]:
        # WNBA 2026 regular season started May 8. Use May 8 to avoid
        # counting pre-season games played in early May.
        start   = datetime.datetime(yr, 5, 8)
        end     = min(datetime.datetime(yr, 10, 1), datetime.datetime.today())
        current = start
        found   = False
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            try:
                resp   = requests.get(f"{ESPN_BASE}/scoreboard",
                                      params={"dates": date_str}, timeout=8)
                events = resp.json().get("events", [])
                for ev in events:
                    status = ev.get("status", {}).get("type", {}).get("name", "")
                    if status != "STATUS_FINAL":
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
                    if h_score == 0 and a_score == 0 or not h_abb or not a_abb:
                        continue
                    found = True
                    own_map.setdefault(h_abb, []).append(h_score)
                    opp_map.setdefault(h_abb, []).append(a_score)
                    own_map.setdefault(a_abb, []).append(a_score)
                    opp_map.setdefault(a_abb, []).append(h_score)
                    if h_score > a_score:
                        wins_map[h_abb] = wins_map.get(h_abb, 0) + 1
                        loss_map[a_abb] = loss_map.get(a_abb, 0) + 1
                    else:
                        wins_map[a_abb] = wins_map.get(a_abb, 0) + 1
                        loss_map[h_abb] = loss_map.get(h_abb, 0) + 1
            except Exception:
                pass
            current += datetime.timedelta(days=1)
        if found:
            break

    result = {}
    for abb in set(list(opp_map.keys()) + list(own_map.keys())):
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


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_todays_games() -> tuple[list, str]:
    """Fetch today's WNBA games from ESPN."""
    today = datetime.datetime.today().strftime("%Y%m%d")
    url   = f"{ESPN_BASE}/scoreboard"
    try:
        resp  = requests.get(url, params={"dates": today}, timeout=10)
        data  = resp.json()
        games = data.get("events", [])
        return games, ""
    except Exception as e:
        return [], str(e)




@st.cache_data(show_spinner=False, ttl=1800)
def fetch_recent_stats(season: int, last_n: int = 5) -> dict:
    """
    Compute per-team averages from their last N completed games.
    Returns {abb: {pts_pg, opp_pts_pg, margin_pg, fg_pct, fg3_pct, ft_pct,
                   efg_pct, reb_pg, ast_pg, tov_pg, stl_pg, blk_pg,
                   off_reb_pg, def_reb_pg, G}}.
    """
    # Collect all completed games per team, most recent first
    games_by_team = {}   # abb -> list of (date, own_score, opp_score, game dict)

    for yr in [season, season - 1]:
        start   = datetime.datetime(yr, 5, 8)
        end     = min(datetime.datetime(yr, 10, 1), datetime.datetime.today())
        # Build day list and scan newest first so we can stop early
        days = []
        cur  = start
        while cur <= end:
            days.append(cur)
            cur += datetime.timedelta(days=1)
        days.reverse()   # newest first

        found_any = False
        for day in days:
            date_str = day.strftime("%Y%m%d")
            try:
                resp   = requests.get(f"{ESPN_BASE}/scoreboard",
                                      params={"dates": date_str}, timeout=8)
                events = resp.json().get("events", [])
                for ev in events:
                    status = ev.get("status", {}).get("type", {}).get("name", "")
                    if status != "STATUS_FINAL":
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
                    if h_score == 0 and a_score == 0 or not h_abb or not a_abb:
                        continue
                    found_any = True
                    # Get per-team box stats from linescores if available
                    for abb, own, opp in [(h_abb, h_score, a_score), (a_abb, a_score, h_score)]:
                        if abb:
                            games_by_team.setdefault(abb, []).append({
                                "date":  day.strftime("%Y-%m-%d"),
                                "own":   own,
                                "opp":   opp,
                            })
            except Exception:
                pass

        if found_any:
            break

    # For each team, take the last N games and average the scores
    # (Box stats like FG% are not available from scoreboard alone —
    #  we approximate recent form using pts/opp_pts which we do have,
    #  and use season averages for shooting %, rebounds, etc.)
    result = {}
    for abb, game_list in games_by_team.items():
        recent = game_list[:last_n]   # already newest-first
        if not recent:
            continue
        n       = len(recent)
        pts_pg  = round(sum(g["own"] for g in recent) / n, 1)
        opp_pg  = round(sum(g["opp"] for g in recent) / n, 1)
        margin  = round(pts_pg - opp_pg, 2)
        w       = sum(1 for g in recent if g["own"] > g["opp"])
        l       = n - w
        result[abb] = {
            "G":          n,
            "pts_pg":     pts_pg,
            "opp_pts_pg": opp_pg,
            "margin_pg":  margin,
            "W":          w,
            "L":          l,
            "W_PCT":      w / n if n > 0 else 0.5,
        }
    return result

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_season_games(season: int) -> tuple[list, str]:
    """Fetch all completed WNBA regular-season games for the given season."""
    all_games = []
    # WNBA season runs May–September
    start = datetime.datetime(season, 5, 8)  # Regular season starts May 8
    end   = min(datetime.datetime(season, 10, 15), datetime.datetime.today())
    url   = f"{ESPN_BASE}/scoreboard"

    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        try:
            resp  = requests.get(url, params={"dates": date_str}, timeout=8)
            events = resp.json().get("events", [])
            for ev in events:
                status = ev.get("status", {}).get("type", {}).get("name", "")
                if status == "STATUS_FINAL":
                    all_games.append(ev)
        except Exception:
            pass
        current += datetime.timedelta(days=1)

    return all_games, ""


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
        "wpct_score":   min(1.0, max(0.0, s.get("wpct", 50) / 100)),
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


def calc_prob(sa, sb, home, w_off, w_def, w_rec,
              ra=None, rb=None, w_season=1.0, w_recent=0.0):
    ea = blend_stats(sa, ra, w_season, w_recent) if ra else sa
    eb = blend_stats(sb, rb, w_season, w_recent) if rb else sb
    off_a, def_a, rec_a, off_b, def_b, rec_b = build_composite(ea, eb)
    sc_a = off_a * w_off + def_a * w_def + rec_a * w_rec
    sc_b = off_b * w_off + def_b * w_def + rec_b * w_rec
    if home == "home":   sc_a += HOME_BOOST
    elif home == "away": sc_b += HOME_BOOST
    total = sc_a + sc_b or 1
    return sc_a / total, sc_b / total


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
    prob_strength = "strong" if prob_gap >= 12 else ("moderate" if prob_gap >= 5 else "narrow")

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

    # Save snapshot once per day — no longer requires standings,
    # since ESPN may not populate them early in the season
    today_str = datetime.datetime.today().strftime("%Y-%m-%d")
    if team_stats_data:
        save_snapshot(team_stats_data, standings_data or {}, today_str)
        if os.path.exists(snapshot_path(today_str)):
            st.caption("📸 Today's snapshot saved ✅")
        else:
            st.caption("📸 Snapshot save failed — check folder permissions")

    st.markdown("---")
    available_snaps = list_snapshots()
    if available_snaps:
        st.caption(f"📸 {len(available_snaps)} snapshot(s) saved  \n"
                   f"Earliest: {available_snaps[0]}  \n"
                   f"Latest: {available_snaps[-1]}")
    else:
        st.caption("📸 No snapshots yet — one will be saved each day the app runs.")


# ── Main tabs ────────────────────────────────────────────────────────────────────
tab_today, tab_back = st.tabs(["📅 Today's Games", "📊 Backtest"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Today's Games
# ═══════════════════════════════════════════════════════════════════════════════
with tab_today:
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

            if sa["G"] < 1 or sb["G"] < 1:
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
                    st.subheader(f"{away_label if (away_label := game['away_name']) else ''} @ {home_label}")
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Backtest
# ═══════════════════════════════════════════════════════════════════════════════
with tab_back:
    st.header("Backtest — Season Results")
    st.caption(
        "Evaluates win-probability and spread picks against actual game outcomes. "
        "Games with a pre-game snapshot use stats as of that snapshot date (no look-ahead bias)."
    )

    num_games = st.slider("Games to evaluate (most recent first)", 10, 200, 60, step=10)

    with st.spinner("Fetching season games..."):
        season_games_raw, err2 = fetch_season_games(SEASON)

    if err2:
        st.warning(f"API issue: {err2}")

    season_games = [g for g in [parse_espn_game(ev) for ev in season_games_raw] if g]
    season_games.sort(key=lambda g: g["date"], reverse=True)

    if not season_games:
        st.info("No completed games found yet this season.")
    else:
        available_snaps  = list_snapshots()
        results_table    = []
        correct_prob     = 0
        covered          = 0
        margin_errors    = []
        total            = 0
        snap_hits        = 0
        snap_misses      = 0

        for game in season_games[:num_games]:
            home_abb = game["home_abb"]
            away_abb = game["away_abb"]

            if not (home_abb in ABB_TO_FULL and away_abb in ABB_TO_FULL):
                continue

            # Use pre-game snapshot if available
            snap_ts, snap_st, snap_date = get_best_snapshot_for_game(game["date"])
            if snap_ts:
                use_ts  = snap_ts
                use_st  = snap_st or standings_data
                snap_hits += 1
                used_snap = snap_date
            else:
                use_ts  = team_stats_data
                use_st  = standings_data
                snap_misses += 1
                used_snap = None

            sa = score_team(home_abb, use_ts, use_st)
            sb = score_team(away_abb, use_ts, use_st)
            if sa["G"] < 1 or sb["G"] < 1:
                continue

            af = arena_factor(home_abb)
            pct_h, pct_a = calc_prob(sa, sb, "home", w_off, w_def, w_rec)
            pct_h_i = round(pct_h * 100)
            pct_a_i = round(pct_a * 100)

            prob_winner = sa["name"] if pct_h >= pct_a else sb["name"]

            proj_h, proj_a, margin_winner, margin, total_pts = calc_spread(
                sa, sb, "home", w_off, w_def, w_rec, af)

            actual_h = game["home_score"]
            actual_a = game["away_score"]
            if actual_h == 0 and actual_a == 0:
                continue

            actual_winner = sa["name"] if actual_h > actual_a else sb["name"]
            actual_margin = actual_h - actual_a   # positive = home win
            proj_margin   = round(proj_h - proj_a, 1)

            # Spread pick: snap to nearest 0.5; fav must win by MORE than the spread
            spread_val  = round(round(abs(margin) * 2) / 2, 1)
            if margin_winner == sa["name"]:
                did_cover = actual_margin > spread_val   # e.g. spread -10.5 → need win by 11+
            else:
                did_cover = actual_margin < -spread_val

            prob_correct = prob_winner == actual_winner
            margin_err   = abs(proj_margin - actual_margin)

            conf_l, conf_e, _, _ = calc_confidence(
                sa, sb, pct_h_i, pct_a_i, margin_winner, prob_winner)

            cover_str = (
                f"✅ **{margin_winner}** covered"
                if did_cover else
                f"❌ **{dog_spread_name if (dog_spread_name := sb['name'] if margin_winner == sa['name'] else sa['name']) else ''}** beat the spread"
            )

            if prob_correct: correct_prob += 1
            if did_cover:    covered      += 1
            margin_errors.append(margin_err)
            total += 1

            results_table.append({
                "date":          game["date"],
                "matchup":       f"{sb['name']} @ {sa['name']}",
                "actual":        f"{sa['name']} {actual_h} – {sb['name']} {actual_a}",
                "actual_margin": actual_margin,
                "prob_pick":     prob_winner,
                "prob_pct":      max(pct_h_i, pct_a_i),
                "prob_correct":  prob_correct,
                "margin_pick":   margin_winner,
                "proj_margin":   proj_margin,
                "did_cover":     did_cover,
                "cover_str":     cover_str,
                "margin_err":    round(margin_err, 1),
                "confidence":    f"{conf_e} {conf_l}",
                "conf_level":    conf_l,
                "used_snapshot": used_snap,
            })

        if total > 0:
            acc_prob       = round(correct_prob / total * 100)
            cover_rate     = round(covered / total * 100)
            avg_margin_err = round(sum(margin_errors) / len(margin_errors), 1) if margin_errors else 0

            if snap_hits + snap_misses > 0:
                snap_pct = round(snap_hits / (snap_hits + snap_misses) * 100)
                if snap_pct == 100:
                    st.success(f"✅ All {total} games evaluated using pre-game snapshots — no look-ahead bias.")
                elif snap_pct > 0:
                    st.info(
                        f"📸 {snap_hits}/{snap_hits+snap_misses} games ({snap_pct}%) used pre-game snapshots. "
                        f"{snap_misses} earlier games used current stats."
                    )
                else:
                    if available_snaps:
                        st.info(
                            f"📸 {len(available_snaps)} snapshot(s) saved from {available_snaps[0]} onward, "
                            f"but all backtested games predate the earliest snapshot. "
                            f"Snapshots will be used going forward."
                        )
                    else:
                        st.warning(
                            "⚠️ No snapshots yet — all games evaluated using current stats. "
                            "The app saves a snapshot each day it runs."
                        )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games evaluated", total)
            m2.metric("Win prob accuracy",  f"{acc_prob}%",       f"{correct_prob}/{total} correct")
            m3.metric("Spread cover rate",  f"{cover_rate}%",     f"{covered}/{total} covered")
            m4.metric("Avg spread error",   f"{avg_margin_err} pts")

            # Accuracy by confidence tier
            st.markdown('<p class="section-head">Accuracy by confidence tier</p>',
                        unsafe_allow_html=True)
            tier_order  = ["High", "Moderate", "Low", "Conflicted"]
            tier_colors = {"High": "#f60", "Moderate": "#f5c842",
                           "Low": "#f5a623", "Conflicted": "#ff5252"}
            tier_emoji  = {"High": "🟢", "Moderate": "🟡", "Low": "🟠", "Conflicted": "🔴"}
            tier_stats  = {t: {"prob_hit": 0, "covered": 0, "total": 0, "errs": []}
                           for t in tier_order}
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
                    ts = tier_stats[tier]
                    pp = round(ts["prob_hit"] / ts["total"] * 100) if ts["total"] else 0
                    cp = round(ts["covered"]  / ts["total"] * 100) if ts["total"] else 0
                    ae = round(sum(ts["errs"]) / len(ts["errs"]), 1) if ts["errs"] else 0
                    c  = tier_colors[tier]
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
                                    color:{'#f60' if pp>=50 else '#ff5252'}">{pp}%</span>
                                <span style="color:#666;font-size:11px;"> ({ts['prob_hit']}/{ts['total']})</span>
                            </div>
                            <div style="font-size:13px;margin-bottom:4px;">
                                <span style="color:#888;">Covered </span>
                                <span style="font-weight:800;font-size:18px;
                                    color:{'#f60' if cp>=50 else '#ff5252'}">{cp}%</span>
                                <span style="color:#666;font-size:11px;"> ({ts['covered']}/{ts['total']})</span>
                            </div>
                            <div style="font-size:11px;color:#888;">Avg error: <span style="color:#ccc;">{ae} pts</span></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("")
                tier_labels = [f"{tier_emoji[t]} {t} ({tier_stats[t]['total']}g)"
                               for t in active_tiers]
                prob_accs   = [round(tier_stats[t]["prob_hit"]/tier_stats[t]["total"]*100)
                               for t in active_tiers]
                cov_rates   = [round(tier_stats[t]["covered"] /tier_stats[t]["total"]*100)
                               for t in active_tiers]

                fig_tier = go.Figure()
                fig_tier.add_trace(go.Bar(
                    name="Win prob accuracy", x=tier_labels, y=prob_accs,
                    marker_color="#f60", opacity=0.85,
                    text=[f"{v}%" for v in prob_accs], textposition="outside",
                    textfont=dict(color="#ccc", size=11)))
                fig_tier.add_trace(go.Bar(
                    name="Spread cover rate", x=tier_labels, y=cov_rates,
                    marker_color="#3d8bff", opacity=0.85,
                    text=[f"{v}%" for v in cov_rates], textposition="outside",
                    textfont=dict(color="#ccc", size=11)))
                fig_tier.add_hline(y=50, line_dash="dot",
                    line_color="rgba(255,255,255,0.25)",
                    annotation_text="50% baseline",
                    annotation_font_color="rgba(255,255,255,0.4)",
                    annotation_position="right")
                fig_tier.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="group", height=280,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1,
                                font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
                    yaxis=dict(range=[0, 120], ticksuffix="%",
                               tickfont=dict(color="#888", size=10),
                               showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                               zeroline=False),
                    xaxis=dict(tickfont=dict(color="#ccc", size=11)),
                )
                st.plotly_chart(fig_tier, use_container_width=True)

            # Overall accuracy bar
            st.markdown('<p class="section-head">Overall accuracy</p>',
                        unsafe_allow_html=True)
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=["Win Probability Model", "Spread Cover Rate"],
                y=[acc_prob, cover_rate],
                marker_color=["#f60" if acc_prob >= 50 else "#ff5252",
                              "#3d8bff" if cover_rate >= 50 else "#ff5252"],
                text=[f"{acc_prob}%", f"{cover_rate}%"],
                textposition="outside", textfont=dict(color="#ccc", size=13),
            ))
            fig_acc.add_hline(y=50, line_dash="dot",
                line_color="rgba(255,255,255,0.3)",
                annotation_text="50% baseline",
                annotation_font_color="rgba(255,255,255,0.4)",
                annotation_position="right")
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                yaxis=dict(range=[0, 120], ticksuffix="%",
                           tickfont=dict(color="#888", size=10),
                           showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                           zeroline=False),
                xaxis=dict(tickfont=dict(color="#ccc", size=12)),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # Game-by-game table
            st.markdown('<p class="section-head">Most recent 25 games</p>',
                        unsafe_allow_html=True)
            st.caption(f"Showing 25 of {total} games. Metrics above reflect all evaluated games.")
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
                        st.caption(
                            f"Spread pick: {row['margin_pick']} · "
                            f"Proj margin: {row['proj_margin']:+.1f} · "
                            f"Actual: {row['actual_margin']:+d} pts")
                    with c5:
                        st.markdown(row["confidence"])
                    st.divider()
        else:
            st.info("Not enough completed games to evaluate yet.")

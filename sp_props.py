"""
MLB Starting Pitcher Props — Strikeout & Performance Model
-----------------------------------------------------------
Install dependencies:
    pip install MLB-StatsAPI streamlit plotly numpy

Run:
    streamlit run sp_props.py
"""

import datetime
import numpy as np
import plotly.graph_objects as go
import statsapi
import streamlit as st

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

def api_name_to_abb(name: str):
    return FULL_TO_ABB.get(STATSAPI_NAME_MAP.get(name, name))


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
    """Full season stats for a pitcher."""
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
        k9   = float(s.get("strikeoutsPer9Inn", "0") or 0)
        bb9  = float(s.get("walksPer9Inn",      "0") or 0)
        era  = float(s.get("era",               "0") or 0)
        whip = float(s.get("whip",              "0") or 0)
        ip   = float(s.get("inningsPitched",    "0") or 0)
        gs   = int(s.get("gamesStarted", 0))
        hr9  = float(s.get("homeRunsPer9",      "0") or 0)
        so   = int(s.get("strikeOuts", 0))
        fip  = round((13 * hr9 + 3 * bb9 - 2 * k9) + 3.2, 2) if k9 else era
        ip_per_start = round(ip / gs, 2) if gs > 0 else 0
        k_per_start  = round(so / gs, 1) if gs > 0 else 0
        return {
            "k9":          k9,
            "bb9":         bb9,
            "era":         era,
            "whip":        whip,
            "fip":         fip,
            "hr9":         hr9,
            "ip":          ip,
            "gs":          gs,
            "so":          so,
            "ip_per_start": ip_per_start,
            "k_per_start":  k_per_start,
            "wins":        int(s.get("wins",   0)),
            "losses":      int(s.get("losses", 0)),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_last_n_starts(player_id: int, season: int, n: int = 5) -> list[dict]:
    """
    Fetch game log for a pitcher and return their last N starts.
    Each entry has: date, opponent, ip, k, er, h, bb.
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
        # Filter to starts only (gamesStarted == 1)
        starts = [sp for sp in splits if int(sp.get("stat", {}).get("gamesStarted", 0)) == 1]
        # Sort by date descending, take last N
        starts = sorted(starts, key=lambda x: x.get("date", ""), reverse=True)[:n]
        result = []
        for sp in starts:
            s = sp.get("stat", {})
            result.append({
                "date":     sp.get("date", "")[:10],
                "opponent": sp.get("opponent", {}).get("name", "?"),
                "ip":       float(s.get("inningsPitched", 0) or 0),
                "k":        int(s.get("strikeOuts", 0)),
                "er":       int(s.get("earnedRuns", 0)),
                "h":        int(s.get("hits", 0)),
                "bb":       int(s.get("baseOnBalls", 0)),
                "hr":       int(s.get("homeRuns", 0)),
            })
        return result
    except Exception:
        return []


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
    """Convenience wrapper returning (season_stats, last_5_starts)."""
    return (
        fetch_pitcher_season_stats(player_id, season),
        fetch_pitcher_last_n_starts(player_id, season, 5),
    )


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_id_by_name(name: str) -> int | None:
    try:
        res = statsapi.lookup_player(name)
        return res[0]["id"] if res else None
    except Exception:
        return None


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
        recent_er_avg = None
        recent_k_avg  = season_k_per_start

    # Blend season and recent K/9
    blended_k9 = round(
        season_k9 * (1 - n_recent_weight) + recent_k9 * n_recent_weight, 2
    )

    # Opposing lineup K% adjustment
    # If opponent K% is 5pp above league avg (0.275 vs 0.225), pitcher gets a boost
    opp_k_pct    = opp_batting.get("k_pct", LEAGUE_K_PCT)
    k_pct_delta  = opp_k_pct - LEAGUE_K_PCT          # positive = more Ks for pitcher
    k9_adj       = round(blended_k9 * (1 + k_pct_delta * 2.5), 2)  # scale factor

    # Expected innings — blend season avg with recent, cap at 7.0
    exp_ip = min(7.0, round(
        season_ip_start * (1 - n_recent_weight) + recent_ip_avg * n_recent_weight, 1
    ))

    # Projected Ks
    proj_k_raw = k9_adj * (exp_ip / 9)
    proj_k     = round(round(proj_k_raw * 2) / 2, 1)  # snap to 0.5

    # Projected runs allowed
    opp_ops      = opp_batting.get("ops",    0.720)
    opp_runs_pg  = opp_batting.get("runs_pg", 4.5)
    park_factor  = PARK_RUN_FACTOR.get(park_abb, 1.00)

    # ERA-based expected ER per 9 innings, adjusted for park and opponent
    pitcher_era = pitcher.get("era", LEAGUE_ERA)
    opp_ops_adj = (opp_ops - 0.720) * 3.0           # ops deviation → run adjustment
    era_adj     = pitcher_era * park_factor + opp_ops_adj
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
today_label = datetime.datetime.today().strftime("%A, %B %d").replace(" 0", " ")
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
all_pitchers = []

progress = st.progress(0, text="Loading pitcher data...")
total_games = len(todays_games)

for i, game in enumerate(todays_games):
    progress.progress((i + 1) / total_games,
                      text=f"Loading {game.get('home_name','?')} vs {game.get('away_name','?')}...")

    h_name = game.get("home_name", "")
    a_name = game.get("away_name", "")
    h_abb  = api_name_to_abb(h_name)
    a_abb  = api_name_to_abb(a_name)

    home_sp_name = game.get("home_probable_pitcher", "TBD")
    away_sp_name = game.get("away_probable_pitcher", "TBD")
    home_sp_pid  = game.get("home_pitcher_id")
    away_sp_pid  = game.get("away_pitcher_id")
    venue_abb    = h_abb or "LAD"
    venue_name   = game.get("venue_name", "")

    # Fetch team batting for each side (opposing lineup)
    h_team_id = game.get("home_id")
    a_team_id = game.get("away_id")
    h_batting = fetch_team_batting(h_team_id, SEASON) if h_team_id else {}
    a_batting = fetch_team_batting(a_team_id, SEASON) if a_team_id else {}

    # Process each SP in this game
    for sp_name, sp_pid, is_home, opp_batting, opp_name, opp_abb in [
        (home_sp_name, home_sp_pid, True,  a_batting, a_name, a_abb),
        (away_sp_name, away_sp_pid, False, h_batting, h_name, h_abb),
    ]:
        if not sp_name or sp_name == "TBD":
            continue

        # Get pitcher ID
        pid = sp_pid
        if not isinstance(pid, int) or not pid:
            pid = fetch_pitcher_id_by_name(sp_name)
        if not pid:
            continue

        season_stats, recent_starts = fetch_pitcher_by_id(pid, SEASON)
        if not season_stats:
            continue

        gs = season_stats.get("gs", 0)
        if gs < min_gs_filter:
            continue

        proj = project_strikeouts(
            season_stats, recent_starts, opp_batting, venue_abb, recent_weight
        )
        if not proj:
            continue

        tier_label, tier_color = quality_tier(season_stats.get("k9", 0))

        all_pitchers.append({
            "name":          sp_name,
            "team":          h_name if is_home else a_name,
            "team_abb":      h_abb  if is_home else a_abb,
            "is_home":       is_home,
            "opp":           opp_name,
            "opp_abb":       opp_abb,
            "venue":         venue_name,
            "venue_abb":     venue_abb,
            "season_stats":  season_stats,
            "recent_starts": recent_starts,
            "proj":          proj,
            "tier_label":    tier_label,
            "tier_color":    tier_color,
            "game_time":     game.get("game_datetime", ""),
        })

progress.empty()

if not all_pitchers:
    st.info("No starting pitcher data available for today's games yet. "
            "Probable starters are usually posted a few hours before game time.")
    st.stop()

# Sort by projected Ks descending
all_pitchers.sort(key=lambda x: x["proj"].get("proj_k", 0), reverse=True)

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

    # Game time
    try:
        gt = datetime.datetime.strptime(p["game_time"], "%Y-%m-%dT%H:%M:%SZ")
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
    # Best K prop recommendation (highest edge with a clear over/under)
    best_line, best_rec, best_col, best_reason = None, None, "#888", ""
    for line in reversed(COMMON_K_LINES):  # check from highest line down
        rec, col, reason = proj["k_props"][line]
        if rec == "OVER":
            best_line, best_rec, best_col, best_reason = line, "OVER", col, reason
            break
    if not best_line:
        for line in COMMON_K_LINES:        # check from lowest line up for UNDER
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

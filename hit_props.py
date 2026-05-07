"""
MLB Batter Hit Probability — Daily App
----------------------------------------
Predicts the probability of each starting batter recording at least one hit today.

Install dependencies:
    pip install MLB-StatsAPI streamlit plotly numpy

Run:
    streamlit run hit_props.py
"""

import datetime
import numpy as np
import plotly.graph_objects as go
import statsapi
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hit Props", page_icon="🥊", layout="wide")

st.markdown("""
<style>
    .section-head {
        font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
        color: #888; margin: 1.2rem 0 0.4rem;
    }
    .prob-high   { color: #00c07a; font-weight: 800; font-size: 2rem; }
    .prob-mid    { color: #f5c842; font-weight: 800; font-size: 2rem; }
    .prob-low    { color: #ff5252; font-weight: 800; font-size: 2rem; }
    .tag {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 11px; font-weight: 600; margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

SEASON     = datetime.datetime.now().year

# ── League-average baselines ───────────────────────────────────────────────────
LEAGUE_AVG_BA    = 0.243   # MLB batting average 2024
LEAGUE_PA_9INN   = 38      # plate appearances per team per 9 innings (league avg)
LEAGUE_ERA       = 4.20

# Expected plate appearances by lineup position (based on historical MLB data)
# Leadoff hitters get ~4.5 PA/game, 9-hole ~3.5 PA/game
LINEUP_PA = {
    1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 3.9,
    6: 3.8, 7: 3.7, 8: 3.6, 9: 3.5,
}

# Ballpark hit factors (slight variation from run factors — some parks suppress
# HR but allow more singles, e.g. Dodger Stadium)
PARK_HIT_FACTOR = {
    "COL": 1.12, "BOS": 1.06, "CHC": 1.04, "TEX": 1.04,
    "CIN": 1.03, "BAL": 1.03, "MIN": 1.02, "DET": 1.02,
    "PHI": 1.02, "ATL": 1.01, "NYY": 1.01, "HOU": 1.01,
    "MIL": 1.01, "ARI": 1.00, "LAD": 1.00, "STL": 1.00,
    "WSN": 1.00, "PIT": 0.99, "TOR": 0.99, "NYM": 0.99,
    "KCR": 0.99, "CLE": 0.98, "CHW": 0.98, "SFG": 0.98,
    "SEA": 0.97, "LAA": 0.97, "SDP": 0.97, "MIA": 0.96,
    "OAK": 0.96, "TBR": 0.95,
}

# ── Team name lookups ──────────────────────────────────────────────────────────
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
def fetch_lineup(game_id: int) -> tuple[list, list]:
    """
    Fetch confirmed starting lineups for a game.
    Returns (home_lineup, away_lineup) where each is a list of player dicts:
    [{id, name, position, batting_order}, ...]
    Returns empty lists if lineups not yet posted.
    """
    try:
        data = statsapi.get("game", {"gamePk": game_id})
        box  = data.get("liveData", {}).get("boxscore", {})
        home_players = box.get("teams", {}).get("home", {}).get("players", {})
        away_players = box.get("teams", {}).get("away", {}).get("players", {})

        def extract_lineup(players: dict) -> list:
            starters = []
            for pid, info in players.items():
                order = info.get("battingOrder")
                if order and str(order).strip():
                    try:
                        order_int = int(str(order).strip()) // 100
                        if 1 <= order_int <= 9:
                            starters.append({
                                "id":            info["person"]["id"],
                                "name":          info["person"]["fullName"],
                                "batting_order": order_int,
                                "position":      info.get("position", {}).get("abbreviation", ""),
                            })
                    except (ValueError, KeyError):
                        pass
            return sorted(starters, key=lambda x: x["batting_order"])

        return extract_lineup(home_players), extract_lineup(away_players)
    except Exception:
        return [], []


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_batter_season_stats(player_id: int, season: int) -> dict:
    """Season batting stats for a hitter."""
    try:
        raw = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=hitting,type=season,season={season})",
        })
        people = raw.get("people", [])
        if not people:
            return {}
        splits = people[0].get("stats", [{}])[0].get("splits", [])
        s = splits[0].get("stat", {}) if splits else {}
        if not s:
            return {}
        ab  = max(int(s.get("atBats", 0) or 0), 1)
        pa  = max(int(s.get("plateAppearances", 0) or 0), 1)
        h   = int(s.get("hits", 0) or 0)
        bb  = int(s.get("baseOnBalls", 0) or 0)
        so  = int(s.get("strikeOuts", 0) or 0)
        return {
            "avg":    float(s.get("avg",  "0") or 0),
            "obp":    float(s.get("obp",  "0") or 0),
            "slg":    float(s.get("slg",  "0") or 0),
            "ops":    float(s.get("ops",  "0") or 0),
            "hits":   h,
            "ab":     ab,
            "pa":     pa,
            "bb_pct": round(bb / pa, 3),
            "k_pct":  round(so / pa, 3),
            "babip":  float(s.get("babip", "0") or 0),
            "hr":     int(s.get("homeRuns", 0) or 0),
            "games":  int(s.get("gamesPlayed", 0) or 0),
            "bats":   people[0].get("batSide", {}).get("code", "R"),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_batter_recent_stats(player_id: int, season: int, last_n_games: int = 15) -> dict:
    """
    Fetch batter's stats over their last N games via game log.
    More reliable than a date-range window since it counts actual games played.
    """
    try:
        raw = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=hitting,type=gameLog,season={season})",
        })
        people = raw.get("people", [])
        if not people:
            return {}
        splits = people[0].get("stats", [{}])[0].get("splits", [])
        # Sort by date descending, take last N
        splits = sorted(splits, key=lambda x: x.get("date", ""), reverse=True)[:last_n_games]
        if not splits:
            return {}
        h  = sum(int(sp["stat"].get("hits",      0) or 0) for sp in splits)
        ab = sum(int(sp["stat"].get("atBats",     0) or 0) for sp in splits)
        pa = sum(int(sp["stat"].get("plateAppearances", 0) or 0) for sp in splits)
        so = sum(int(sp["stat"].get("strikeOuts", 0) or 0) for sp in splits)
        # Games with at least one hit
        games_with_hit = sum(
            1 for sp in splits if int(sp["stat"].get("hits", 0) or 0) >= 1
        )
        return {
            "avg":            round(h / ab, 3) if ab > 0 else 0,
            "hit_game_rate":  round(games_with_hit / len(splits), 3),
            "games":          len(splits),
            "hits":           h,
            "ab":             ab,
            "k_pct":          round(so / pa, 3) if pa > 0 else 0,
            "games_with_hit": games_with_hit,
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_batter_vs_pitcher(batter_id: int, pitcher_id: int) -> dict:
    """Historical matchup stats between a specific batter and pitcher."""
    try:
        raw = statsapi.get("people", {
            "personIds": batter_id,
            "hydrate": f"stats(group=hitting,type=vsPlayer,opposingPlayerId={pitcher_id})",
        })
        people = raw.get("people", [])
        if not people:
            return {}
        splits = people[0].get("stats", [{}])[0].get("splits", [])
        s = splits[0].get("stat", {}) if splits else {}
        if not s:
            return {}
        ab = int(s.get("atBats", 0) or 0)
        h  = int(s.get("hits",   0) or 0)
        return {
            "avg": round(h / ab, 3) if ab > 0 else 0,
            "ab":  ab,
            "h":   h,
            "hr":  int(s.get("homeRuns", 0) or 0),
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_batter_splits(player_id: int, season: int) -> dict:
    """
    Fetch platoon splits — batter's avg vs LHP and vs RHP.
    """
    splits_data = {"vs_L": {}, "vs_R": {}}
    for split_type, key in [("vsleft", "vs_L"), ("vsright", "vs_R")]:
        try:
            raw = statsapi.get("people", {
                "personIds": player_id,
                "hydrate": f"stats(group=hitting,type={split_type},season={season})",
            })
            people = raw.get("people", [])
            if not people:
                continue
            splits = people[0].get("stats", [{}])[0].get("splits", [])
            s = splits[0].get("stat", {}) if splits else {}
            if not s:
                continue
            ab = int(s.get("atBats", 0) or 0)
            splits_data[key] = {
                "avg": float(s.get("avg", "0") or 0),
                "ab":  ab,
                "ops": float(s.get("ops", "0") or 0),
            }
        except Exception:
            pass
    return splits_data


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_handedness(pitcher_id: int) -> str:
    """Return 'L' or 'R' for a pitcher's throwing hand."""
    try:
        raw = statsapi.get("people", {"personIds": pitcher_id})
        people = raw.get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code", "R")
    except Exception:
        pass
    return "R"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pitcher_stats(pitcher_id: int, season: int) -> dict:
    """Season stats for the opposing starting pitcher."""
    try:
        raw = statsapi.get("people", {
            "personIds": pitcher_id,
            "hydrate": f"stats(group=pitching,type=season,season={season})",
        })
        people = raw.get("people", [])
        if not people:
            return {}
        splits = people[0].get("stats", [{}])[0].get("splits", [])
        s = splits[0].get("stat", {}) if splits else {}
        if not s:
            return {}
        ip = float(s.get("inningsPitched", "0") or 0)
        gs = int(s.get("gamesStarted", 0) or 0)
        return {
            "era":        float(s.get("era",  "0") or 0),
            "whip":       float(s.get("whip", "0") or 0),
            "avg_against":float(s.get("avg",  "0") or 0),  # batting avg against
            "k9":         float(s.get("strikeoutsPer9Inn", "0") or 0),
            "ip":         ip,
            "gs":         gs,
            "ip_per_start": round(ip / gs, 1) if gs > 0 else 0,
        }
    except Exception:
        return {}


# ── Core model ─────────────────────────────────────────────────────────────────
def project_hit_probability(
    season_stats:  dict,
    recent_stats:  dict,
    splits:        dict,
    vs_pitcher:    dict,
    pitcher_stats: dict,
    pitcher_hand:  str,
    lineup_pos:    int,
    park_abb:      str,
    w_season:      float = 0.5,
    w_recent:      float = 0.5,
    recent_games:  int   = 15,
) -> dict:
    """
    Project the probability that a batter records at least 1 hit.

    Core formula:
        P(≥1 hit) = 1 - (1 - adjusted_hit_rate) ^ expected_PA

    adjusted_hit_rate accounts for:
        - Season BA (blended with recent BA)
        - Recent game-by-game hit rate
        - Platoon split vs today's SP hand
        - Career vs this pitcher (if 10+ AB)
        - Opposing SP quality (avg against)
        - Ballpark hit factor
    """
    if not season_stats:
        return {}

    season_avg = season_stats.get("avg", LEAGUE_AVG_BA) or LEAGUE_AVG_BA
    recent_avg = recent_stats.get("avg", season_avg)    or season_avg
    recent_hit_game_rate = recent_stats.get("hit_game_rate", 0)
    recent_g   = recent_stats.get("games", 0)

    # ── Step 1: Base hit rate — blend season and recent BA ────────────────────
    if recent_g >= 5:
        base_avg = season_avg * w_season + recent_avg * w_recent
    else:
        base_avg = season_avg  # not enough recent data, trust season

    # ── Step 2: Platoon split adjustment ──────────────────────────────────────
    batter_hand   = season_stats.get("bats", "R")
    platoon_key   = "vs_L" if pitcher_hand == "L" else "vs_R"
    platoon_data  = splits.get(platoon_key, {})
    platoon_avg   = platoon_data.get("avg", 0)
    platoon_ab    = platoon_data.get("ab",  0)
    platoon_adj   = 0.0

    if platoon_ab >= 20:
        # Platoon split is meaningful — blend it in (30% weight)
        platoon_adj = (platoon_avg - base_avg) * 0.30

    # ── Step 3: Career vs this pitcher ────────────────────────────────────────
    matchup_avg = vs_pitcher.get("avg", 0)
    matchup_ab  = vs_pitcher.get("ab",  0)
    matchup_adj = 0.0

    if matchup_ab >= 10:
        # Meaningful sample — weight by AB (more AB = more trust)
        matchup_weight = min(0.25, matchup_ab / 100)
        matchup_adj = (matchup_avg - base_avg) * matchup_weight

    # ── Step 4: Opposing SP quality adjustment ────────────────────────────────
    sp_avg_against = pitcher_stats.get("avg_against", LEAGUE_AVG_BA)
    sp_ip          = pitcher_stats.get("ip", 0)
    sp_adj         = 0.0

    if sp_ip >= 10:
        # Pitcher's BA against reflects how hard it is to get hits off them
        # If their avg_against is .220 vs league .243, batters get a penalty
        sp_adj = (sp_avg_against - LEAGUE_AVG_BA) * 0.40

    # ── Step 5: Combine adjustments ───────────────────────────────────────────
    adjusted_avg = base_avg + platoon_adj + matchup_adj + sp_adj

    # ── Step 6: Ballpark hit factor ───────────────────────────────────────────
    park_factor  = PARK_HIT_FACTOR.get(park_abb, 1.00)
    adjusted_avg = adjusted_avg * park_factor

    # Clip to realistic range
    adjusted_avg = max(0.10, min(0.420, adjusted_avg))

    # ── Step 7: Expected plate appearances ────────────────────────────────────
    exp_pa = LINEUP_PA.get(lineup_pos, 3.8)
    # Adjust PA slightly for SP expected innings (shorter outings = more bullpen)
    sp_ip_per_start = pitcher_stats.get("ip_per_start", 5.5)
    if sp_ip_per_start < 5.0:
        exp_pa *= 0.95  # slightly fewer PA vs quick-hook SP

    # ── Step 8: P(at least 1 hit) ─────────────────────────────────────────────
    p_no_hit  = (1 - adjusted_avg) ** exp_pa
    p_hit     = round((1 - p_no_hit) * 100, 1)

    # ── Step 9: Confidence ────────────────────────────────────────────────────
    data_points = (
        (1 if season_stats.get("games", 0) >= 20 else 0) +
        (1 if recent_g >= 8 else 0) +
        (1 if platoon_ab >= 20 else 0) +
        (1 if matchup_ab >= 10 else 0) +
        (1 if sp_ip >= 10 else 0)
    )
    if data_points >= 4:   conf = "High"
    elif data_points >= 2: conf = "Moderate"
    else:                  conf = "Low"

    # ── Step 10: Key reasons ──────────────────────────────────────────────────
    reasons = []

    if recent_g >= 5:
        trend = "🔥 Hot" if recent_avg > season_avg + 0.020 else (
                "❄️ Cold" if recent_avg < season_avg - 0.020 else "➡️ On pace")
        reasons.append(f"{trend} — last {recent_g} games: .{int(recent_avg*1000):03d} "
                       f"({recent_stats.get('games_with_hit',0)}/{recent_g} games with a hit)")

    if platoon_ab >= 20:
        adv = "advantage" if platoon_adj > 0.005 else (
              "disadvantage" if platoon_adj < -0.005 else "neutral")
        reasons.append(f"Platoon {adv} vs {'LHP' if pitcher_hand=='L' else 'RHP'} "
                       f"(.{int(platoon_avg*1000):03d} in {platoon_ab} AB)")

    if matchup_ab >= 10:
        reasons.append(f"Career vs this pitcher: .{int(matchup_avg*1000):03d} "
                       f"({vs_pitcher.get('h',0)}-for-{matchup_ab})")

    if sp_ip >= 10:
        sp_label = "tough" if sp_adj < -0.010 else ("hittable" if sp_adj > 0.010 else "average")
        reasons.append(f"Opposing SP is {sp_label} (.{int(sp_avg_against*1000):03d} avg against)")

    if park_factor >= 1.04:
        reasons.append(f"Hitter-friendly park ({park_abb}, +{round((park_factor-1)*100)}% hit factor)")
    elif park_factor <= 0.97:
        reasons.append(f"Pitcher-friendly park ({park_abb}, {round((park_factor-1)*100)}% hit factor)")

    return {
        "p_hit":           p_hit,
        "adjusted_avg":    round(adjusted_avg, 3),
        "base_avg":        round(base_avg, 3),
        "exp_pa":          round(exp_pa, 1),
        "platoon_adj":     round(platoon_adj, 3),
        "matchup_adj":     round(matchup_adj, 3),
        "sp_adj":          round(sp_adj, 3),
        "park_factor":     park_factor,
        "pitcher_hand":    pitcher_hand,
        "confidence":      conf,
        "reasons":         reasons,
        "recent_hit_rate": recent_hit_game_rate,
        "recent_games":    recent_g,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🥊 Hit Props Model")
    st.caption(f"Season: {SEASON} · statsapi")

    st.markdown("---")
    st.markdown("##### Model settings")

    recent_games_n = st.select_slider(
        "Recent form window (last N games)",
        options=[10, 15, 20, 25], value=15,
        help="How many recent games to use for form analysis."
    )
    w_recent_pct = st.slider(
        "Recent form weight", 0, 100, 40, step=10,
        help="How much to weight recent BA vs season BA. 40% = balanced."
    )
    w_season = (100 - w_recent_pct) / 100
    w_recent = w_recent_pct / 100

    st.markdown("---")
    st.markdown("##### Filters")
    min_prob    = st.slider("Min hit probability to show", 50, 85, 60, step=5)
    conf_filter = st.multiselect(
        "Confidence levels",
        ["High", "Moderate", "Low"],
        default=["High", "Moderate"],
    )
    show_reasons = st.checkbox("Show reasoning", value=True)

    st.markdown("---")
    st.caption(
        "P(hit) = 1 − (1 − adj_avg)^PA\n\n"
        "Adjustments: recent form, platoon split, career vs SP, SP quality, park factor."
    )


# ── Main panel ─────────────────────────────────────────────────────────────────
today_label = datetime.datetime.today().strftime("%A, %B %d").replace(" 0", " ")
st.markdown(f"## 🥊 Hit Props — {today_label}")
st.caption(
    "Probability of each starting batter recording at least one hit today. "
    "Lineups must be posted for projections to appear (usually 3–4 hours before first pitch)."
)

with st.spinner("Fetching today's schedule..."):
    games, err = fetch_todays_games()

if err:
    st.error(f"Schedule error: {err}")
    st.stop()
if not games:
    st.info("No MLB regular season games today.")
    st.stop()

# ── Load all lineups and build projections ─────────────────────────────────────
all_batters   = []
games_no_lineup = 0
progress = st.progress(0, text="Loading lineups and stats...")

for gi, game in enumerate(games):
    progress.progress((gi + 1) / len(games),
                      text=f"Loading {game.get('away_name','?')} @ {game.get('home_name','?')}...")

    game_id   = game["game_id"]
    home_name = game.get("home_name", "")
    away_name = game.get("away_name", "")
    home_abb  = api_name_to_abb(home_name) or "LAD"
    away_abb  = api_name_to_abb(away_name) or "LAD"

    # Game time
    try:
        gt = datetime.datetime.strptime(game.get("game_datetime", ""), "%Y-%m-%dT%H:%M:%SZ")
        time_str = gt.strftime("%I:%M %p UTC").lstrip("0")
    except Exception:
        time_str = ""

    # SP info
    home_sp_name = game.get("home_probable_pitcher", "TBD")
    away_sp_name = game.get("away_probable_pitcher", "TBD")
    home_sp_id   = game.get("home_pitcher_id")
    away_sp_id   = game.get("away_pitcher_id")

    # Fetch SP stats and handedness
    home_sp_stats, home_sp_hand = {}, "R"
    away_sp_stats, away_sp_hand = {}, "R"

    if isinstance(home_sp_id, int) and home_sp_id:
        home_sp_stats = fetch_pitcher_stats(home_sp_id, SEASON)
        home_sp_hand  = fetch_pitcher_handedness(home_sp_id)
    elif home_sp_name and home_sp_name != "TBD":
        try:
            res = statsapi.lookup_player(home_sp_name)
            if res:
                home_sp_id    = res[0]["id"]
                home_sp_stats = fetch_pitcher_stats(home_sp_id, SEASON)
                home_sp_hand  = fetch_pitcher_handedness(home_sp_id)
        except Exception:
            pass

    if isinstance(away_sp_id, int) and away_sp_id:
        away_sp_stats = fetch_pitcher_stats(away_sp_id, SEASON)
        away_sp_hand  = fetch_pitcher_handedness(away_sp_id)
    elif away_sp_name and away_sp_name != "TBD":
        try:
            res = statsapi.lookup_player(away_sp_name)
            if res:
                away_sp_id    = res[0]["id"]
                away_sp_stats = fetch_pitcher_stats(away_sp_id, SEASON)
                away_sp_hand  = fetch_pitcher_handedness(away_sp_id)
        except Exception:
            pass

    # Fetch lineups
    home_lineup, away_lineup = fetch_lineup(game_id)

    if not home_lineup and not away_lineup:
        games_no_lineup += 1
        continue

    # Process each batter
    for batter, is_home, park_abb, opp_sp_id, opp_sp_stats, opp_sp_hand, opp_sp_name, opp_name in [
        # Home batters face away SP
        *[(b, True,  home_abb, away_sp_id, away_sp_stats, away_sp_hand, away_sp_name, away_name)
          for b in home_lineup],
        # Away batters face home SP
        *[(b, False, away_abb, home_sp_id, home_sp_stats, home_sp_hand, home_sp_name, home_name)
          for b in away_lineup],
    ]:
        pid     = batter["id"]
        b_name  = batter["name"]
        bat_ord = batter["batting_order"]

        season_s = fetch_batter_season_stats(pid, SEASON)
        if not season_s or season_s.get("games", 0) < 3:
            continue

        recent_s = fetch_batter_recent_stats(pid, SEASON, recent_games_n)
        splits_s = fetch_batter_splits(pid, SEASON)

        # Career vs pitcher
        vs_sp = {}
        if isinstance(opp_sp_id, int) and opp_sp_id:
            vs_sp = fetch_batter_vs_pitcher(pid, opp_sp_id)

        proj = project_hit_probability(
            season_stats  = season_s,
            recent_stats  = recent_s,
            splits        = splits_s,
            vs_pitcher    = vs_sp,
            pitcher_stats = opp_sp_stats,
            pitcher_hand  = opp_sp_hand,
            lineup_pos    = bat_ord,
            park_abb      = park_abb,
            w_season      = w_season,
            w_recent      = w_recent,
            recent_games  = recent_games_n,
        )
        if not proj:
            continue

        team_name = home_name if is_home else away_name
        team_abb  = home_abb  if is_home else away_abb

        all_batters.append({
            "name":        b_name,
            "team":        team_name,
            "team_abb":    team_abb,
            "is_home":     is_home,
            "opp":         opp_name,
            "opp_sp":      opp_sp_name,
            "opp_sp_hand": opp_sp_hand,
            "bat_order":   bat_ord,
            "park_abb":    park_abb,
            "game_time":   time_str,
            "season_avg":  season_s.get("avg", 0),
            "season_ops":  season_s.get("ops", 0),
            "season_games":season_s.get("games", 0),
            "proj":        proj,
        })

progress.empty()

if games_no_lineup > 0:
    st.info(
        f"⏳ Lineups not yet posted for {games_no_lineup} game(s). "
        f"Check back closer to first pitch (usually 3–4 hours before game time)."
    )

if not all_batters:
    st.warning("No lineup data available yet for today's games. Check back later!")
    st.stop()

# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = [
    b for b in all_batters
    if b["proj"]["p_hit"] >= min_prob
    and b["proj"]["confidence"] in conf_filter
]
filtered.sort(key=lambda x: x["proj"]["p_hit"], reverse=True)

st.caption(
    f"{len(filtered)} batters shown (of {len(all_batters)} total) · "
    f"min {min_prob}% hit probability · "
    f"recent form: last {recent_games_n} games at {w_recent_pct}% weight"
)
st.markdown("")

if not filtered:
    st.info(f"No batters meet the current filters. Try lowering the minimum probability "
            f"or including more confidence levels.")
    st.stop()


# ── Render batter cards ────────────────────────────────────────────────────────
conf_colors = {"High": "#00c07a", "Moderate": "#f5c842", "Low": "#f5a623"}
conf_emoji  = {"High": "🟢",     "Moderate": "🟡",     "Low": "🟠"}

for b in filtered:
    proj  = b["proj"]
    p     = proj["p_hit"]
    conf  = proj["confidence"]
    color = conf_colors.get(conf, "#888")

    prob_class = "prob-high" if p >= 75 else ("prob-mid" if p >= 65 else "prob-low")

    ha_str   = "🏠" if b["is_home"] else "✈️"
    hand_str = f"vs {'LHP' if b['opp_sp_hand']=='L' else 'RHP'} {b['opp_sp']}"

    with st.container():
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;'
            f'border-left:4px solid {color};border-radius:12px;padding:16px 20px;margin-bottom:12px;">',
            unsafe_allow_html=True,
        )

        # Header row
        c_name, c_prob, c_meta = st.columns([3, 1, 2])
        with c_name:
            st.markdown(
                f'<div style="font-size:18px;font-weight:800;">{b["name"]}</div>'
                f'<div style="font-size:12px;color:#888;margin-top:2px;">'
                f'{ha_str} {b["team"]} · #{b["bat_order"]} · {hand_str} · {b["game_time"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c_prob:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div class="{prob_class}">{p}%</div>'
                f'<div style="font-size:11px;color:#888;">hit prob</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c_meta:
            st.markdown(
                f'<div style="text-align:right;">'
                f'<div style="font-size:12px;color:{color};font-weight:700;">'
                f'{conf_emoji[conf]} {conf} confidence</div>'
                f'<div style="font-size:11px;color:#888;margin-top:4px;">'
                f'Adj avg: .{int(proj["adjusted_avg"]*1000):03d} · '
                f'{proj["exp_pa"]} exp PA</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Stats row
        st.markdown('<p class="section-head">Key stats</p>', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1: st.metric("Season AVG", f".{int(b['season_avg']*1000):03d}")
        with s2: st.metric("Season OPS", f"{b['season_ops']:.3f}")
        with s3:
            if proj["recent_games"] >= 5:
                rg = proj["recent_games"]
                rhr = proj["recent_hit_rate"]
                st.metric(f"Last {rg}G hit rate",
                          f"{round(rhr*100)}%",
                          f"{round(b['proj']['recent_hit_rate']*rg)}/{rg} games")
            else:
                st.metric("Recent form", "< 5G data")
        with s4:
            adj_delta = round((proj["adjusted_avg"] - proj["base_avg"]) * 1000)
            st.metric("Adj AVG", f".{int(proj['adjusted_avg']*1000):03d}",
                      delta=f"{'+' if adj_delta >= 0 else ''}{adj_delta} pts",
                      help="After platoon, matchup, SP, and park adjustments")
        with s5:
            pf = proj["park_factor"]
            pf_str = f"{'+' if pf >= 1 else ''}{round((pf-1)*100, 0):.0f}%"
            st.metric("Park factor", pf_str, b["park_abb"])

        # Adjustment breakdown mini chart
        adjs = {
            "Base avg":       proj["base_avg"],
            "Platoon":        proj["platoon_adj"],
            "vs SP history":  proj["matchup_adj"],
            "SP quality":     proj["sp_adj"],
            "Park":           round((proj["park_factor"] - 1) * proj["base_avg"], 3),
        }
        non_base = {k: v for k, v in adjs.items() if k != "Base avg" and v != 0}
        if non_base:
            fig = go.Figure()
            colors_bar = ["#00c07a" if v >= 0 else "#ff5252" for v in non_base.values()]
            fig.add_trace(go.Bar(
                x=list(non_base.values()),
                y=list(non_base.keys()),
                orientation="h",
                marker_color=colors_bar,
                text=[f"{'+' if v>=0 else ''}{round(v*1000):+d} pts" for v in non_base.values()],
                textposition="outside",
                textfont=dict(color="#ccc", size=10),
            ))
            fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=120, margin=dict(l=10, r=80, t=5, b=5),
                xaxis=dict(showgrid=False, zeroline=False,
                           tickfont=dict(color="#888", size=9)),
                yaxis=dict(tickfont=dict(color="#aaa", size=10)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                               key=f"adj_chart_{b['name']}_{b['team_abb']}_{b['bat_order']}")

        # Reasons
        if show_reasons and proj["reasons"]:
            for reason in proj["reasons"]:
                st.markdown(
                    f'<div style="font-size:12px;color:#aaa;margin:2px 0;">• {reason}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")


# ── Summary table ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Quick Reference — All Qualifying Batters")
st.caption(f"All {len(filtered)} batters meeting your filters, ranked by hit probability.")

hdr = ["Batter", "Team", "Ord", "vs SP", "Season AVG", "Adj AVG", "Exp PA", "Hit Prob", "Conf"]
header_html = "".join(
    f"<th style='padding:6px 10px;color:#888;font-size:10px;letter-spacing:1.5px;"
    f"text-transform:uppercase;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1);'>{h}</th>"
    for h in hdr
)

rows_html = ""
for b in filtered:
    proj  = b["proj"]
    p     = proj["p_hit"]
    conf  = proj["confidence"]
    color = conf_colors.get(conf, "#888")
    p_color = "#00c07a" if p >= 75 else ("#f5c842" if p >= 65 else "#ff5252")
    rows_html += (
        f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
        f"<td style='padding:8px 10px;color:#ccc;font-weight:600;'>{b['name']}</td>"
        f"<td style='padding:8px 10px;color:#888;font-size:12px;'>{b['team']}</td>"
        f"<td style='padding:8px 10px;color:#888;text-align:center;'>#{b['bat_order']}</td>"
        f"<td style='padding:8px 10px;color:#aaa;font-size:11px;'>"
        f"{'LHP' if b['opp_sp_hand']=='L' else 'RHP'} {b['opp_sp']}</td>"
        f"<td style='padding:8px 10px;color:#ccc;'>.{int(b['season_avg']*1000):03d}</td>"
        f"<td style='padding:8px 10px;color:#ccc;'>.{int(proj['adjusted_avg']*1000):03d}</td>"
        f"<td style='padding:8px 10px;color:#aaa;'>{proj['exp_pa']}</td>"
        f"<td style='padding:8px 10px;font-weight:800;font-size:16px;color:{p_color};'>{p}%</td>"
        f"<td style='padding:8px 10px;font-weight:700;color:{color};'>"
        f"{conf_emoji[conf]} {conf}</td>"
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
    "⚠️ Projections are statistical estimates only. Lineup changes, weather, and "
    "in-game decisions are not modeled. For entertainment purposes only."
)

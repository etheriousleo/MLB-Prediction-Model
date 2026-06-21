"""
FIFA World Cup 2026 — Match Predictor & Tournament Simulator
-------------------------------------------------------------
Data: Built-in team ratings derived from FIFA Rankings (April 2026),
      historical World Cup performance, and Elo-based attack/defense estimates.
      Live scores are fetched from ESPN's public soccer scoreboard API.

Install:
    pip install streamlit plotly numpy requests

Run:
    streamlit run wc2026_app.py
"""

import datetime
import json
import math
import os
import random
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup 2026",
    page_icon="🏆",
    layout="wide",
)

st.markdown("""
<style>
    .big-prob { font-size: 3rem; font-weight: 800; line-height: 1.1; }
    .win-green { color: #00c07a; }
    .win-draw  { color: #f0b429; }
    .win-red   { color: #ff5252; }
    .winner-box {
        background: linear-gradient(135deg, #1a221a, #0d1a1a);
        border: 1px solid #00c07a55;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin: 0.75rem 0;
    }
    .group-card {
        background: #0e1a1e;
        border: 1px solid #1e3040;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .tier-elite    { color: #ffd700; font-weight: 700; }
    .tier-strong   { color: #00c07a; font-weight: 700; }
    .tier-solid    { color: #4fc3f7; font-weight: 700; }
    .tier-longshot { color: #aaaaaa; font-weight: 700; }
    .conf-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-left: 6px;
    }
    .section-head {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        margin: 1rem 0 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEAM DATA — FIFA rankings, Elo-derived attack/defense, historical WC record
# ══════════════════════════════════════════════════════════════════════════════

# fmt: off
TEAMS = {
    # name: {fifa_rank, elo, attack, defense, wc_wins, group, flag, confederation}
    # attack/defense on 0-10 scale; elo ~1400-2000
    # Sources: FIFA ranking April 2026, club-adjusted Elo estimates
    "France":           {"fifa_rank":1,  "elo":1877, "attack":8.8,"defense":8.6,"wc_wins":2,"group":"I","flag":"🇫🇷","conf":"UEFA"},
    "Spain":            {"fifa_rank":2,  "elo":1876, "attack":8.7,"defense":8.5,"wc_wins":1,"group":"H","flag":"🇪🇸","conf":"UEFA"},
    "Argentina":        {"fifa_rank":3,  "elo":1875, "attack":8.7,"defense":8.2,"wc_wins":3,"group":"J","flag":"🇦🇷","conf":"CONMEBOL"},
    "England":          {"fifa_rank":4,  "elo":1826, "attack":8.4,"defense":8.2,"wc_wins":1,"group":"L","flag":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","conf":"UEFA"},
    "Portugal":         {"fifa_rank":5,  "elo":1764, "attack":8.5,"defense":7.8,"wc_wins":0,"group":"K","flag":"🇵🇹","conf":"UEFA"},
    "Brazil":           {"fifa_rank":6,  "elo":1761, "attack":8.3,"defense":7.9,"wc_wins":5,"group":"C","flag":"🇧🇷","conf":"CONMEBOL"},
    "Netherlands":      {"fifa_rank":7,  "elo":1758, "attack":8.1,"defense":7.8,"wc_wins":0,"group":"F","flag":"🇳🇱","conf":"UEFA"},
    "Morocco":          {"fifa_rank":8,  "elo":1756, "attack":7.5,"defense":8.4,"wc_wins":0,"group":"C","flag":"🇲🇦","conf":"CAF"},
    "Belgium":          {"fifa_rank":9,  "elo":1735, "attack":8.0,"defense":7.7,"wc_wins":0,"group":"G","flag":"🇧🇪","conf":"UEFA"},
    "Germany":          {"fifa_rank":10, "elo":1730, "attack":7.9,"defense":7.7,"wc_wins":4,"group":"E","flag":"🇩🇪","conf":"UEFA"},
    "Croatia":          {"fifa_rank":11, "elo":1717, "attack":7.6,"defense":7.8,"wc_wins":0,"group":"L","flag":"🇭🇷","conf":"UEFA"},
    "Colombia":         {"fifa_rank":13, "elo":1693, "attack":7.5,"defense":7.3,"wc_wins":0,"group":"K","flag":"🇨🇴","conf":"CONMEBOL"},
    "Senegal":          {"fifa_rank":14, "elo":1689, "attack":7.3,"defense":7.5,"wc_wins":0,"group":"I","flag":"🇸🇳","conf":"CAF"},
    "Mexico":           {"fifa_rank":15, "elo":1681, "attack":7.2,"defense":7.2,"wc_wins":0,"group":"A","flag":"🇲🇽","conf":"CONCACAF"},
    "United States":    {"fifa_rank":16, "elo":1673, "attack":7.1,"defense":7.1,"wc_wins":0,"group":"D","flag":"🇺🇸","conf":"CONCACAF"},
    "Uruguay":          {"fifa_rank":17, "elo":1673, "attack":7.3,"defense":7.4,"wc_wins":2,"group":"H","flag":"🇺🇾","conf":"CONMEBOL"},
    "Japan":            {"fifa_rank":18, "elo":1660, "attack":7.2,"defense":7.0,"wc_wins":0,"group":"F","flag":"🇯🇵","conf":"AFC"},
    "Switzerland":      {"fifa_rank":19, "elo":1649, "attack":6.9,"defense":7.4,"wc_wins":0,"group":"B","flag":"🇨🇭","conf":"UEFA"},
    "Australia":        {"fifa_rank":26, "elo":1580, "attack":6.4,"defense":6.5,"wc_wins":0,"group":"D","flag":"🇦🇺","conf":"AFC"},
    "South Korea":      {"fifa_rank":22, "elo":1603, "attack":6.7,"defense":6.6,"wc_wins":0,"group":"A","flag":"🇰🇷","conf":"AFC"},
    "Ecuador":          {"fifa_rank":23, "elo":1595, "attack":6.6,"defense":6.5,"wc_wins":0,"group":"E","flag":"🇪🇨","conf":"CONMEBOL"},
    "Austria":          {"fifa_rank":24, "elo":1594, "attack":6.8,"defense":6.7,"wc_wins":0,"group":"J","flag":"🇦🇹","conf":"UEFA"},
    "Türkiye":          {"fifa_rank":25, "elo":1589, "attack":6.7,"defense":6.6,"wc_wins":0,"group":"D","flag":"🇹🇷","conf":"UEFA"},
    "Canada":           {"fifa_rank":27, "elo":1565, "attack":6.5,"defense":6.5,"wc_wins":0,"group":"B","flag":"🇨🇦","conf":"CONCACAF"},
    "Norway":           {"fifa_rank":29, "elo":1554, "attack":6.9,"defense":6.3,"wc_wins":0,"group":"I","flag":"🇳🇴","conf":"UEFA"},
    "Algeria":          {"fifa_rank":34, "elo":1530, "attack":6.4,"defense":6.3,"wc_wins":0,"group":"J","flag":"🇩🇿","conf":"CAF"},
    "Egypt":            {"fifa_rank":35, "elo":1524, "attack":6.3,"defense":6.3,"wc_wins":0,"group":"G","flag":"🇪🇬","conf":"CAF"},
    "Scotland":         {"fifa_rank":36, "elo":1520, "attack":6.2,"defense":6.2,"wc_wins":0,"group":"C","flag":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","conf":"UEFA"},
    "Paraguay":         {"fifa_rank":39, "elo":1510, "attack":6.1,"defense":6.3,"wc_wins":0,"group":"D","flag":"🇵🇾","conf":"CONMEBOL"},
    "Tunisia":          {"fifa_rank":41, "elo":1498, "attack":6.0,"defense":6.2,"wc_wins":0,"group":"F","flag":"🇹🇳","conf":"CAF"},
    "Ivory Coast":      {"fifa_rank":42, "elo":1494, "attack":6.4,"defense":5.9,"wc_wins":0,"group":"E","flag":"🇨🇮","conf":"CAF"},
    "Sweden":           {"fifa_rank":43, "elo":1490, "attack":6.3,"defense":6.3,"wc_wins":0,"group":"F","flag":"🇸🇪","conf":"UEFA"},
    "Czechia":          {"fifa_rank":44, "elo":1488, "attack":6.0,"defense":6.3,"wc_wins":0,"group":"A","flag":"🇨🇿","conf":"UEFA"},
    "Qatar":            {"fifa_rank":53, "elo":1445, "attack":5.6,"defense":5.8,"wc_wins":0,"group":"B","flag":"🇶🇦","conf":"AFC"},
    "Saudi Arabia":     {"fifa_rank":60, "elo":1430, "attack":5.8,"defense":5.6,"wc_wins":0,"group":"H","flag":"🇸🇦","conf":"AFC"},
    "South Africa":     {"fifa_rank":61, "elo":1428, "attack":5.5,"defense":5.7,"wc_wins":0,"group":"A","flag":"🇿🇦","conf":"CAF"},
    "Bosnia & Herz.":   {"fifa_rank":65, "elo":1420, "attack":5.8,"defense":5.6,"wc_wins":0,"group":"B","flag":"🇧🇦","conf":"UEFA"},
    "Jordan":           {"fifa_rank":64, "elo":1422, "attack":5.6,"defense":5.7,"wc_wins":0,"group":"J","flag":"🇯🇴","conf":"AFC"},
    "Cape Verde":       {"fifa_rank":67, "elo":1416, "attack":5.7,"defense":5.5,"wc_wins":0,"group":"H","flag":"🇨🇻","conf":"CAF"},
    "Ghana":            {"fifa_rank":74, "elo":1400, "attack":5.8,"defense":5.4,"wc_wins":0,"group":"L","flag":"🇬🇭","conf":"CAF"},
    "Curaçao":          {"fifa_rank":82, "elo":1382, "attack":5.3,"defense":5.1,"wc_wins":0,"group":"E","flag":"🇨🇼","conf":"CONCACAF"},
    "Haiti":            {"fifa_rank":83, "elo":1380, "attack":5.2,"defense":5.0,"wc_wins":0,"group":"C","flag":"🇭🇹","conf":"CONCACAF"},
    "New Zealand":      {"fifa_rank":85, "elo":1374, "attack":5.1,"defense":5.2,"wc_wins":0,"group":"G","flag":"🇳🇿","conf":"OFC"},
    "Iran":             {"fifa_rank":20, "elo":1618, "attack":6.5,"defense":6.9,"wc_wins":0,"group":"G","flag":"🇮🇷","conf":"AFC"},
    "Iraq":             {"fifa_rank":65, "elo":1420, "attack":5.7,"defense":5.5,"wc_wins":0,"group":"I","flag":"🇮🇶","conf":"AFC"},
    "DR Congo":         {"fifa_rank":55, "elo":1440, "attack":5.9,"defense":5.7,"wc_wins":0,"group":"K","flag":"🇨🇩","conf":"CAF"},
    "Uzbekistan":       {"fifa_rank":50, "elo":1460, "attack":6.0,"defense":5.9,"wc_wins":0,"group":"K","flag":"🇺🇿","conf":"AFC"},
    "Panama":           {"fifa_rank":30, "elo":1550, "attack":6.0,"defense":6.4,"wc_wins":0,"group":"L","flag":"🇵🇦","conf":"CONCACAF"},
}
# fmt: on

# Clean up duplicate Senegal key (dict keeps last) — already handled by insertion order
# Rebuild with deduplication just in case
_seen = set()
_TEAMS_CLEAN = {}
for k, v in TEAMS.items():
    if k not in _seen:
        _TEAMS_CLEAN[k] = v
        _seen.add(k)
TEAMS = _TEAMS_CLEAN

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia & Herz."],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

CONF_COLORS = {
    "UEFA":    "#4a90d9",
    "CONMEBOL":"#2ecc71",
    "CAF":     "#e67e22",
    "AFC":     "#e74c3c",
    "CONCACAF":"#9b59b6",
    "OFC":     "#1abc9c",
}

# ── Snapshot system ──────────────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wc2026_snapshots")

def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def snapshot_path(date_str):
    return os.path.join(SNAPSHOT_DIR, f"ratings_{date_str}.json")

def save_snapshot(ratings, date_str=None):
    ensure_snapshot_dir()
    if date_str is None:
        date_str = datetime.datetime.today().strftime("%Y-%m-%d")
    path = snapshot_path(date_str)
    if os.path.exists(path):
        return
    try:
        with open(path, "w") as f:
            json.dump({"date": date_str, "ratings": ratings}, f)
    except Exception:
        pass

def load_snapshot(date_str):
    path = snapshot_path(date_str)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def list_snapshots():
    ensure_snapshot_dir()
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")]
    dates = []
    for fn in sorted(files, reverse=True):
        d = fn.replace("ratings_", "").replace(".json", "")
        dates.append(d)
    return dates

# ── Match data fetcher (openfootball open data) ──────────────────────────────
WC_DATA_URL = "https://raw.githubusercontent.com/openfootball/world-cup.json/master/2026/worldcup.json"

# Normalize team names from openfootball format to our TEAMS dict keys
OFB_NAME_MAP = {
    "Czech Republic":           "Czechia",
    "Bosnia & Herzegovina":     "Bosnia & Herz.",
    "Bosnia and Herzegovina":   "Bosnia & Herz.",
    "USA":                      "United States",
    "Turkey":                   "Türkiye",
    "Curacao":                  "Curaçao",
    "Côte d'Ivoire":           "Ivory Coast",
    "Cote d'Ivoire":           "Ivory Coast",
    "DR Congo":                 "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cabo Verde":               "Cape Verde",
    "Korea Republic":           "South Korea",
}

def normalize_team(name):
    return OFB_NAME_MAP.get(name, name)

@st.cache_data(ttl=300)  # type: ignore[misc]
def fetch_wc_matches():
    """Fetch all 104 World Cup matches from openfootball open data."""
    try:
        r = requests.get(WC_DATA_URL, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        raw_matches = data.get("matches", [])
        matches = []
        today = datetime.date.today()
        for m in raw_matches:
            raw_date = m.get("date", "")
            try:
                match_date = datetime.date.fromisoformat(raw_date)
            except Exception:
                match_date = None

            score = m.get("score", {})
            ft = score.get("ft") if score else None
            ht = score.get("ht") if score else None

            team1_raw = m.get("team1", "")
            team2_raw = m.get("team2", "")

            if not team1_raw or not team2_raw:
                continue

            team1 = normalize_team(team1_raw)
            team2 = normalize_team(team2_raw)

            # A match has known teams iff both sides resolve to real entries. This
            # robustly excludes openfootball bracket placeholders ("1A", "2B",
            # "3A/B/C/D/F", "W73", "L101") regardless of their exact format.
            teams_known = (team1 in TEAMS) and (team2 in TEAMS)

            if match_date:
                if ft:
                    status = "final"
                elif match_date < today:
                    status = "final"
                elif match_date == today:
                    status = "today"
                else:
                    status = "scheduled"
            else:
                status = "scheduled"

            matches.append({
                "date": raw_date,
                "match_date": match_date,
                "time": m.get("time", ""),
                "round": m.get("round", ""),
                "group": m.get("group", ""),
                "team1": team1,
                "team2": team2,
                "team1_raw": team1_raw,
                "team2_raw": team2_raw,
                "score_ft": ft,
                "score_ht": ht,
                "venue": m.get("ground", ""),
                "status": status,
                "teams_known": teams_known,
                "goals1": m.get("goals1", []),
                "goals2": m.get("goals2", []),
            })
        return matches
    except Exception:
        return []

# ── Core Model ───────────────────────────────────────────────────────────────

def completed_group_results(matches):
    """Map group letter -> {(team1, team2): (g1, g2)} for finished group-stage games.

    Lets the group/tournament simulators lock in matches that have already happened
    instead of re-simulating the entire group from scratch. Group letters from the
    feed ("Group A") map straight onto our GROUPS keys ("A"); teams are already
    normalized to TEAMS keys by fetch_wc_matches().
    """
    out = {}
    for m in matches:
        grp = m.get("group", "")
        if not grp.startswith("Group"):
            continue
        if m["status"] != "final" or not m["score_ft"]:
            continue
        if m["team1"] not in TEAMS or m["team2"] not in TEAMS:
            continue
        letter = grp.replace("Group", "").strip()
        g1, g2 = m["score_ft"][0], m["score_ft"][1]
        out.setdefault(letter, {})[(m["team1"], m["team2"])] = (g1, g2)
    return out

def elo_win_prob(elo_a, elo_b, home_advantage=0):
    """Standard Elo win probability with optional home advantage."""
    diff = elo_a - elo_b + home_advantage
    return 1 / (1 + 10 ** (-diff / 400))

def match_probs(team_a, team_b, neutral=True, group_stage=True):
    """
    Return (win_a, draw, win_b) probabilities.
    Uses Elo + attack/defense composite with draw inflation for group stage.
    """
    t_a = TEAMS[team_a]
    t_b = TEAMS[team_b]

    home_adv = 0 if neutral else 80  # Elo points
    p_win_a = elo_win_prob(t_a["elo"], t_b["elo"], home_advantage=home_adv)

    # Attack/defense quality modifier
    atk_edge = (t_a["attack"] - t_b["attack"]) + (t_b["defense"] - t_a["defense"])
    qual_adj = atk_edge * 0.012  # small modifier
    p_win_a = max(0.03, min(0.94, p_win_a + qual_adj))

    # Draw probability — higher in group stage, lower in knockout
    elo_gap = abs(t_a["elo"] - t_b["elo"])
    base_draw = 0.26 if group_stage else 0.0  # no draws in knockout (goes to ET/pens)
    draw_factor = max(0, base_draw - elo_gap * 0.0002)

    p_win_a_adj = p_win_a * (1 - draw_factor)
    p_win_b_adj = (1 - p_win_a) * (1 - draw_factor)
    return round(p_win_a_adj, 4), round(draw_factor, 4), round(p_win_b_adj, 4)

def expected_goals(team_a, team_b):
    """Simple Poisson-parameterized expected goals for each team."""
    t_a = TEAMS[team_a]
    t_b = TEAMS[team_b]
    base_xg = 1.15
    xg_a = base_xg * (t_a["attack"] / 7.0) * (7.0 / t_b["defense"])
    xg_b = base_xg * (t_b["attack"] / 7.0) * (7.0 / t_a["defense"])
    return round(xg_a, 2), round(xg_b, 2)

def _poisson_pmf(k, lam):
    """Exact Poisson probability mass for k goals at rate lam."""
    return math.exp(-lam) * lam ** k / math.factorial(k)

def score_distribution(xg_a, xg_b, max_goals=8):
    """Exact scoreline probabilities from independent Poisson marginals.

    Replaces a 100k-draw Monte Carlo: the joint of two independent Poissons is
    just the product of their PMFs, so this is exact, deterministic, and instant.
    """
    pmf_a = [_poisson_pmf(k, xg_a) for k in range(max_goals + 1)]
    pmf_b = [_poisson_pmf(k, xg_b) for k in range(max_goals + 1)]
    grid = [
        (f"{a}-{b}", pmf_a[a] * pmf_b[b])
        for a in range(max_goals + 1)
        for b in range(max_goals + 1)
    ]
    grid.sort(key=lambda x: x[1], reverse=True)
    return [(s, round(p * 100, 1)) for s, p in grid[:8]]

# ── Shared result helpers (used by group + tournament simulators) ─────────────

def _apply_result(pts, gd, gf, a, b, ga, gb):
    """Accumulate one match result into points / goal-diff / goals-for dicts."""
    gd[a] += ga - gb
    gd[b] += gb - ga
    gf[a] += ga
    gf[b] += gb
    if ga > gb:
        pts[a] += 3
    elif ga == gb:
        pts[a] += 1
        pts[b] += 1
    else:
        pts[b] += 3

def _played_score(played, a, b):
    """Return (a_goals, b_goals) for the a-vs-b matchup if it has been played, else None.

    `played` is keyed by the (team1, team2) order as it appeared in the feed, so we
    check both orientations and normalize the result back to (a_goals, b_goals).
    """
    if (a, b) in played:
        return played[(a, b)]
    if (b, a) in played:
        b_goals, a_goals = played[(b, a)]
        return (a_goals, b_goals)
    return None

def confidence_tier(p_win):
    if p_win >= 0.68:
        return "ELITE", "tier-elite"
    elif p_win >= 0.55:
        return "STRONG", "tier-strong"
    elif p_win >= 0.44:
        return "SOLID", "tier-solid"
    else:
        return "LONGSHOT", "tier-longshot"

# ── Group stage simulation ───────────────────────────────────────────────────

def simulate_group(group_name, played=None, n=50_000):
    """Monte Carlo group stage simulation conditioned on results already played.

    Matches found in `played` are fixed to their real scorelines; only the remaining
    fixtures are simulated. Expected goals for the remaining fixtures are computed once
    up front rather than every iteration.
    """
    teams = GROUPS[group_name]
    played = played or {}
    matchups = [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]

    fixed = []      # (a, b, ga, gb) — already played
    remaining = []  # (a, b, xga, xgb) — to be simulated
    for (a, b) in matchups:
        ps = _played_score(played, a, b)
        if ps is not None:
            fixed.append((a, b, ps[0], ps[1]))
        else:
            xga, xgb = expected_goals(a, b)
            remaining.append((a, b, xga, xgb))

    # If the group is already decided, the standings are deterministic.
    if not remaining:
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        for (a, b, ga, gb) in fixed:
            _apply_result(pts, gd, gf, a, b, ga, gb)
        ranked = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], TEAMS[t]["elo"]), reverse=True)
        adv = {t: (100.0 if t in ranked[:2] else 0.0) for t in teams}
        win = {t: (100.0 if t == ranked[0] else 0.0) for t in teams}
        return adv, win

    advance_counts = {t: 0 for t in teams}
    group_wins = {t: 0 for t in teams}

    for _ in range(n):
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gf_total = {t: 0 for t in teams}

        for (a, b, ga, gb) in fixed:
            _apply_result(pts, gd, gf_total, a, b, ga, gb)
        for (a, b, xga, xgb) in remaining:
            ga = np.random.poisson(xga)
            gb = np.random.poisson(xgb)
            _apply_result(pts, gd, gf_total, a, b, ga, gb)

        # Sort: points → goal diff → goals for → Elo (tiebreaker)
        ranked = sorted(teams,
            key=lambda t: (pts[t], gd[t], gf_total[t], TEAMS[t]["elo"]),
            reverse=True)

        advance_counts[ranked[0]] += 1  # 1st place — automatic
        advance_counts[ranked[1]] += 1  # 2nd place — automatic
        group_wins[ranked[0]] += 1

    return (
        {t: round(advance_counts[t] / n * 100, 1) for t in teams},
        {t: round(group_wins[t] / n * 100, 1) for t in teams},
    )

# ── Full tournament simulator ─────────────────────────────────────────────────

def simulate_tournament(played=None, n=20_000):
    """Full Monte Carlo tournament simulation. Returns champion probability per team.

    The group phase is conditioned on results already played (`played`); only remaining
    group fixtures are simulated. Per-group fixed/remaining splits and expected goals are
    computed once before the iteration loop. The knockout bracket remains a simplified
    re-drawn knockout (see Model Notes).
    """
    played = played or {}
    champ_counts = {t: 0 for t in TEAMS}

    # Precompute each group's locked results + remaining-fixture xG a single time.
    group_plan = {}
    for gname, gteams in GROUPS.items():
        matchups = [(gteams[i], gteams[j]) for i in range(4) for j in range(i + 1, 4)]
        gp = played.get(gname, {})
        fixed, remaining = [], []
        for (a, b) in matchups:
            ps = _played_score(gp, a, b)
            if ps is not None:
                fixed.append((a, b, ps[0], ps[1]))
            else:
                xga, xgb = expected_goals(a, b)
                remaining.append((a, b, xga, xgb))
        group_plan[gname] = (gteams, fixed, remaining)

    for _ in range(n):
        standings = {}
        third_placers = []

        for gname, (gteams, fixed, remaining) in group_plan.items():
            pts = {t: 0 for t in gteams}
            gd = {t: 0 for t in gteams}
            gfs = {t: 0 for t in gteams}

            for (a, b, ga, gb) in fixed:
                _apply_result(pts, gd, gfs, a, b, ga, gb)
            for (a, b, xga, xgb) in remaining:
                ga = np.random.poisson(xga)
                gb = np.random.poisson(xgb)
                _apply_result(pts, gd, gfs, a, b, ga, gb)

            ranked = sorted(gteams,
                key=lambda t: (pts[t], gd[t], gfs[t], TEAMS[t]["elo"]),
                reverse=True)
            standings[gname] = ranked
            third_placers.append((ranked[2], pts[ranked[2]], gd[ranked[2]], gfs[ranked[2]]))

        # Best 8 third-place teams advance under the 48-team format.
        third_placers.sort(key=lambda x: (x[1], x[2], x[3], TEAMS[x[0]]["elo"]), reverse=True)
        best_thirds = [t[0] for t in third_placers[:8]]

        # 32 qualifiers: 12 winners + 12 runners-up + 8 best thirds (all disjoint).
        qualifiers = []
        for gname in sorted(GROUPS):
            qualifiers.append(standings[gname][0])
            qualifiers.append(standings[gname][1])
        qualifiers.extend(best_thirds)

        # Knockout rounds: 32 → 16 → 8 → 4 → 2 → 1 (re-drawn each round).
        bracket = qualifiers[:]
        for _ in range(5):
            if len(bracket) < 2:
                break
            random.shuffle(bracket)
            next_round = []
            for i in range(0, len(bracket) - 1, 2):
                a, b = bracket[i], bracket[i + 1]
                p_a, _, _ = match_probs(a, b, neutral=True, group_stage=False)
                next_round.append(a if random.random() < p_a else b)
            if len(bracket) % 2 == 1:
                next_round.append(bracket[-1])
            bracket = next_round

        if bracket:
            champ_counts[bracket[0]] += 1

    return {t: round(champ_counts[t] / n * 100, 2) for t in TEAMS}

# ══════════════════════════════════════════════════════════════════════════════
# APP UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏆 FIFA World Cup 2026")
st.caption("Match predictor · Group simulator · Tournament odds · Live scores")

tabs = st.tabs(["🎯 Match Predictor", "👥 Group Simulator", "🏆 Tournament Odds", "📺 Live Scores", "📸 Backtest"])

# ══════════════════════════════════════
# TAB 1: MATCH PREDICTOR
# ══════════════════════════════════════
with tabs[0]:
    st.subheader("Match Predictor")
    st.caption("Head-to-head match probability using Elo + attack/defense ratings")

    all_teams = sorted(TEAMS.keys())

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        team_a = st.selectbox("Team A", all_teams,
            index=all_teams.index("France") if "France" in all_teams else 0,
            key="ta")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        neutral = st.checkbox("Neutral Venue", value=True)
        stage = st.radio("Stage", ["Group Stage", "Knockout"], horizontal=True, index=0)
        is_group = (stage == "Group Stage")
    with c3:
        team_b = st.selectbox("Team B", all_teams,
            index=all_teams.index("Argentina") if "Argentina" in all_teams else 1,
            key="tb")

    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        pw_a, pd, pw_b = match_probs(team_a, team_b, neutral=neutral, group_stage=is_group)
        xga, xgb = expected_goals(team_a, team_b)
        top_scores = score_distribution(xga, xgb)
        tier_a, cls_a = confidence_tier(pw_a)
        tier_b, cls_b = confidence_tier(pw_b)
        t_a = TEAMS[team_a]
        t_b = TEAMS[team_b]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='winner-box'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2rem'>{t_a['flag']} {team_a}</div>", unsafe_allow_html=True)
            color_a = "win-green" if pw_a > pw_b else ("win-draw" if pw_a == pw_b else "win-red")
            st.markdown(f"<div class='big-prob {color_a}'>{pw_a*100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='{cls_a}'>⬡ {tier_a}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#888;font-size:0.8rem'>Elo: {t_a['elo']} | FIFA #{t_a['fifa_rank']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if is_group:
                st.markdown(f"<div class='winner-box' style='padding-top:2rem'>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#888;font-size:0.85rem;margin-bottom:0.3rem'>DRAW</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-prob win-draw'>{pd*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='winner-box' style='padding:1rem;text-align:center;color:#888'>Knockout — No draws<br><small>Decided in extra time / pens if level</small></div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='winner-box'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2rem'>{t_b['flag']} {team_b}</div>", unsafe_allow_html=True)
            color_b = "win-green" if pw_b > pw_a else ("win-draw" if pw_a == pw_b else "win-red")
            st.markdown(f"<div class='big-prob {color_b}'>{pw_b*100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='{cls_b}'>⬡ {tier_b}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#888;font-size:0.8rem'>Elo: {t_b['elo']} | FIFA #{t_b['fifa_rank']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # xG and score distribution
        st.markdown("---")
        xc1, xc2 = st.columns(2)
        with xc1:
            st.markdown("<div class='section-head'>Expected Goals</div>", unsafe_allow_html=True)
            fig_xg = go.Figure()
            fig_xg.add_bar(
                x=[f"{t_a['flag']} {team_a}", f"{t_b['flag']} {team_b}"],
                y=[xga, xgb],
                marker_color=["#00c07a", "#4fc3f7"],
                text=[f"xG: {xga}", f"xG: {xgb}"],
                textposition="outside",
            )
            fig_xg.update_layout(
                height=220, margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=False,
                yaxis=dict(gridcolor="#222"),
            )
            st.plotly_chart(fig_xg, use_container_width=True)

        with xc2:
            st.markdown("<div class='section-head'>Most Likely Scorelines</div>", unsafe_allow_html=True)
            score_labels = [s[0] for s in top_scores[:6]]
            score_pcts   = [s[1] for s in top_scores[:6]]
            fig_sc = go.Figure()
            fig_sc.add_bar(
                x=score_labels, y=score_pcts,
                marker_color="#9b59b6",
                text=[f"{p}%" for p in score_pcts],
                textposition="outside",
            )
            fig_sc.update_layout(
                height=220, margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=False,
                yaxis=dict(gridcolor="#222", title="Probability %"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Team comparison radar
        st.markdown("<div class='section-head'>Team Profile Comparison</div>", unsafe_allow_html=True)
        cats = ["Attack", "Defense", "Elo (norm)", "FIFA Rank (inv)", "WC Titles"]
        def norm_elo(e): return (e - 1370) / (1880 - 1370) * 10
        def inv_rank(r): return max(0, (90 - r) / 89 * 10)
        def wc_score(w): return min(10, w * 2)

        vals_a = [
            t_a["attack"], t_a["defense"],
            round(norm_elo(t_a["elo"]), 1),
            round(inv_rank(t_a["fifa_rank"]), 1),
            wc_score(t_a["wc_wins"]),
        ]
        vals_b = [
            t_b["attack"], t_b["defense"],
            round(norm_elo(t_b["elo"]), 1),
            round(inv_rank(t_b["fifa_rank"]), 1),
            wc_score(t_b["wc_wins"]),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_a + [vals_a[0]], theta=cats + [cats[0]],
            fill="toself", name=team_a, line_color="#00c07a", fillcolor="rgba(0,192,122,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]], theta=cats + [cats[0]],
            fill="toself", name=team_b, line_color="#4fc3f7", fillcolor="rgba(79,195,247,0.15)",
        ))
        fig_radar.update_layout(
            height=320,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 10], color="#555"),
                angularaxis=dict(color="#aaa"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Save snapshot
        today_str = datetime.datetime.today().strftime("%Y-%m-%d")
        ratings_snapshot = {t: {"elo": TEAMS[t]["elo"], "fifa_rank": TEAMS[t]["fifa_rank"]} for t in TEAMS}
        save_snapshot(ratings_snapshot, today_str)

# ══════════════════════════════════════
# TAB 2: GROUP SIMULATOR
# ══════════════════════════════════════
with tabs[1]:
    st.subheader("Group Stage Simulator")
    st.caption("Monte Carlo simulation of group standings (50,000 iterations)")

    group_choice = st.selectbox("Select Group", [f"Group {g}" for g in sorted(GROUPS.keys())])
    gkey = group_choice.replace("Group ", "")
    gteams = GROUPS[gkey]

    played_all = completed_group_results(fetch_wc_matches())
    played_grp = played_all.get(gkey, {})
    n_played = len(played_grp)
    if n_played:
        st.caption(f"🔒 {n_played} of 6 matches in Group {gkey} already played — locked into the simulation; "
                   f"the remaining {6 - n_played} are simulated.")

    if st.button("🔄 Run Group Simulation", key="run_group"):
        spin = ("Group already decided…" if n_played == 6
                else f"Simulating {6 - n_played} remaining matches × 50,000 scenarios…")
        with st.spinner(spin):
            adv_pcts, win_pcts = simulate_group(gkey, played=played_grp)
        st.success("Done!")
        st.session_state[f"group_result_{gkey}"] = (adv_pcts, win_pcts)

    if f"group_result_{gkey}" in st.session_state:
        adv_pcts, win_pcts = st.session_state[f"group_result_{gkey}"]

        # Sort by advancement probability
        sorted_teams = sorted(gteams, key=lambda t: adv_pcts[t], reverse=True)

        st.markdown("---")
        gc1, gc2 = st.columns(2)

        with gc1:
            st.markdown("##### Advancement Probability")
            fig_adv = go.Figure()
            colors = ["#ffd700" if i == 0 else "#00c07a" if i == 1 else "#4fc3f7" if adv_pcts[t] > 30 else "#555"
                      for i, t in enumerate(sorted_teams)]
            fig_adv.add_bar(
                x=[f"{TEAMS[t]['flag']} {t}" for t in sorted_teams],
                y=[adv_pcts[t] for t in sorted_teams],
                marker_color=colors,
                text=[f"{adv_pcts[t]}%" for t in sorted_teams],
                textposition="outside",
            )
            fig_adv.update_layout(
                height=280, margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=False,
                yaxis=dict(range=[0, 110], gridcolor="#222", title="%"),
            )
            st.plotly_chart(fig_adv, use_container_width=True)

        with gc2:
            st.markdown("##### Group Winner Probability")
            fig_win = go.Figure()
            fig_win.add_bar(
                x=[f"{TEAMS[t]['flag']} {t}" for t in sorted_teams],
                y=[win_pcts[t] for t in sorted_teams],
                marker_color="#9b59b6",
                text=[f"{win_pcts[t]}%" for t in sorted_teams],
                textposition="outside",
            )
            fig_win.update_layout(
                height=280, margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=False,
                yaxis=dict(range=[0, 110], gridcolor="#222", title="%"),
            )
            st.plotly_chart(fig_win, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Team Details")
        for t in sorted_teams:
            ti = TEAMS[t]
            conf_color = CONF_COLORS.get(ti["conf"], "#888")
            adv_p = adv_pcts[t]
            tier, cls = confidence_tier(adv_p / 100)
            st.markdown(f"""
            <div class='group-card'>
                <b>{ti['flag']} {t}</b>
                <span class='conf-badge' style='background:{conf_color}22;color:{conf_color};border:1px solid {conf_color}55'>{ti['conf']}</span>
                &nbsp;|&nbsp; FIFA #{ti['fifa_rank']} &nbsp;|&nbsp; Elo: {ti['elo']}
                &nbsp;|&nbsp; Atk: <b>{ti['attack']}</b> / Def: <b>{ti['defense']}</b>
                &nbsp;|&nbsp; Advance: <span class='{cls}'>{adv_p}%</span>
                &nbsp;|&nbsp; Win Group: <b>{win_pcts[t]}%</b>
            </div>
            """, unsafe_allow_html=True)

        # Head-to-head matrix
        st.markdown("---")
        st.markdown("##### Head-to-Head Win Probability Matrix")
        matrix_rows = []
        for ta in sorted_teams:
            row = []
            for tb in sorted_teams:
                if ta == tb:
                    row.append(None)
                else:
                    pw_a, _, _ = match_probs(ta, tb, group_stage=True)
                    row.append(pw_a * 100)
            matrix_rows.append(row)

        z_vals = [[0 if v is None else v for v in row] for row in matrix_rows]
        text_vals = [["-" if v is None else f"{v:.1f}%" for v in row] for row in matrix_rows]
        labels = [f"{TEAMS[t]['flag']} {t}" for t in sorted_teams]

        fig_mat = go.Figure(go.Heatmap(
            z=z_vals, x=labels, y=labels,
            text=text_vals, texttemplate="%{text}",
            colorscale=[[0, "#1a1a2e"], [0.4, "#16213e"], [0.6, "#0f3460"], [1.0, "#00c07a"]],
            showscale=False,
        ))
        fig_mat.update_layout(
            height=260, margin=dict(t=10, b=10, l=100, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc", size=11),
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        st.caption("Row team's probability of beating column team in group stage")

    else:
        st.info("Click **Run Group Simulation** to see probabilities.")

    # Show all group compositions
    st.markdown("---")
    st.markdown("##### All Groups Overview")
    for g in sorted(GROUPS.keys()):
        st.markdown(f"**Group {g}**")
        row = st.columns(4)
        for i, t in enumerate(GROUPS[g]):
            ti = TEAMS[t]
            conf_color = CONF_COLORS.get(ti["conf"], "#888")
            row[i].markdown(
                f"<div style='background:#0e1a1e;border:1px solid #1e3040;border-radius:8px;padding:0.5rem;text-align:center'>"
                f"<div style='font-size:1.4rem'>{ti['flag']}</div>"
                f"<div style='font-size:0.75rem;font-weight:600'>{t}</div>"
                f"<div style='font-size:0.65rem;color:{conf_color}'>{ti['conf']}</div>"
                f"<div style='font-size:0.65rem;color:#888'>#{ti['fifa_rank']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("")

# ══════════════════════════════════════
# TAB 3: TOURNAMENT ODDS
# ══════════════════════════════════════
with tabs[2]:
    st.subheader("Tournament Championship Odds")
    st.caption("Full Monte Carlo simulation of all 104 matches (20,000 iterations)")

    n_sims = st.slider("Simulation iterations", 5_000, 30_000, 20_000, step=5_000)

    played_all = completed_group_results(fetch_wc_matches())
    n_locked = sum(len(v) for v in played_all.values())
    if n_locked:
        st.caption(f"🔒 {n_locked} of 72 group matches already played are locked in; "
                   f"remaining group fixtures and the full knockout bracket are simulated.")

    if st.button("🔄 Run Full Tournament Simulation", key="run_tourn"):
        with st.spinner(f"Simulating {n_sims:,} World Cup tournaments… this takes ~15-30 seconds"):
            champ_probs = simulate_tournament(played=played_all, n=n_sims)
        st.session_state["champ_probs"] = champ_probs
        st.success("Done!")

    if "champ_probs" in st.session_state:
        champ_probs = st.session_state["champ_probs"]

        # Top 16 teams
        top_teams = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:16]

        st.markdown("---")
        st.markdown("##### Championship Probability — Top 16")
        fig_champ = go.Figure()
        t_labels = [f"{TEAMS[t]['flag']} {t}" for t, _ in top_teams]
        t_probs  = [p for _, p in top_teams]
        bar_colors = ["#ffd700" if i == 0 else "#c0c0c0" if i == 1 else "#cd7f32" if i == 2
                      else "#00c07a" if p >= 5 else "#4fc3f7" if p >= 2 else "#555"
                      for i, (_, p) in enumerate(top_teams)]
        fig_champ.add_bar(
            x=t_labels, y=t_probs,
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in t_probs],
            textposition="outside",
        )
        fig_champ.update_layout(
            height=380, margin=dict(t=20, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"), showlegend=False,
            yaxis=dict(range=[0, max(t_probs) * 1.2], gridcolor="#222", title="Championship %"),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_champ, use_container_width=True)

        # Confederation breakdown
        st.markdown("---")
        conf_totals = {}
        for t, p in champ_probs.items():
            c = TEAMS[t]["conf"]
            conf_totals[c] = conf_totals.get(c, 0) + p

        st.markdown("##### Championship % by Confederation")
        cf1, cf2, cf3 = st.columns(3)
        conf_sorted = sorted(conf_totals.items(), key=lambda x: x[1], reverse=True)
        for i, (conf, pct) in enumerate(conf_sorted):
            col = [cf1, cf2, cf3][i % 3]
            col_hex = CONF_COLORS.get(conf, "#888")
            col.markdown(
                f"<div style='background:{col_hex}15;border:1px solid {col_hex}44;border-radius:8px;"
                f"padding:0.6rem 1rem;text-align:center;margin-bottom:0.5rem'>"
                f"<div style='color:{col_hex};font-weight:700;font-size:1rem'>{conf}</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:#eee'>{pct:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Full table
        st.markdown("---")
        st.markdown("##### Full 48-Team Championship Odds")
        all_sorted = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)
        for rank_i, (t, p) in enumerate(all_sorted, 1):
            ti = TEAMS[t]
            conf_color = CONF_COLORS.get(ti["conf"], "#888")
            tier, cls = confidence_tier(p / 100)
            medal = "🥇" if rank_i == 1 else "🥈" if rank_i == 2 else "🥉" if rank_i == 3 else f"{rank_i}."
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:0.3rem 0.5rem;"
                f"border-bottom:1px solid #1e2a30;font-size:0.85rem'>"
                f"<span style='width:2rem;color:#888'>{medal}</span>"
                f"<span style='width:1.5rem'>{ti['flag']}</span>"
                f"<span style='flex:1'>{t}</span>"
                f"<span style='color:{conf_color};width:6rem;font-size:0.72rem'>{ti['conf']}</span>"
                f"<span style='width:5rem;text-align:right;font-weight:700'>Group {ti['group']}</span>"
                f"<span style='width:6rem;text-align:right;font-weight:800;color:#00c07a'>{p:.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Click **Run Full Tournament Simulation** to generate championship odds.")


# ══════════════════════════════════════
# TAB 4: LIVE SCORES & RESULTS
# ══════════════════════════════════════
with tabs[3]:
    st.subheader("📺 Live Scores & Results")
    st.caption("Match data from openfootball open data — refreshes every 5 minutes")

    if st.button("🔄 Refresh"):
        st.cache_data.clear()

    all_matches = fetch_wc_matches()

    if not all_matches:
        st.error("Could not fetch match data. Check your internet connection.")
    else:
        today = datetime.date.today()

        final_matches  = [m for m in all_matches if m["status"] == "final" and m["score_ft"]]
        today_matches  = [m for m in all_matches if m["status"] == "today"]
        upcoming       = [m for m in all_matches if m["status"] == "scheduled"]

        # ── TODAY ──
        if today_matches:
            st.markdown("### 📅 Today's Matches")
            for m in today_matches:
                t1 = m["team1"]; t2 = m["team2"]
                f1 = TEAMS.get(t1, {}).get("flag", "🏳"); f2 = TEAMS.get(t2, {}).get("flag", "🏳")
                grp = m["group"] or m["round"]
                pw1, pd, pw2 = match_probs(t1, t2) if t1 in TEAMS and t2 in TEAMS else (None, None, None)
                sc1, sc2, sc3 = st.columns([4, 3, 4])
                with sc1:
                    st.markdown(f"### {f1} {t1}")
                with sc2:
                    st.markdown(f"<div style='text-align:center;padding-top:0.4rem'>"
                                f"<div style='font-size:1.5rem;font-weight:800'>vs</div>"
                                f"<div style='color:#888;font-size:0.75rem'>{grp}</div>"
                                f"<div style='color:#888;font-size:0.75rem'>{m['time']}</div>"
                                f"</div>", unsafe_allow_html=True)
                with sc3:
                    st.markdown(f"### {f2} {t2}")
                if pw1 is not None:
                    pc1, pc2, pc3 = st.columns([4, 3, 4])
                    pc1.markdown(f"<div style='text-align:center;color:#00c07a;font-weight:700'>{pw1*100:.1f}%</div>", unsafe_allow_html=True)
                    pc2.markdown(f"<div style='text-align:center;color:#f0b429;font-weight:700'>Draw {pd*100:.1f}%</div>", unsafe_allow_html=True)
                    pc3.markdown(f"<div style='text-align:center;color:#4fc3f7;font-weight:700'>{pw2*100:.1f}%</div>", unsafe_allow_html=True)
                st.caption(f"📍 {m['venue']}")
                st.markdown("---")

        # ── RECENT RESULTS ──
        if final_matches:
            st.markdown(f"### ✅ Results ({len(final_matches)} matches played)")

            # Group filter
            groups_played = sorted(set(m["group"] for m in final_matches if m["group"]))
            filter_opts = ["All Groups"] + groups_played
            sel_grp = st.selectbox("Filter by group", filter_opts, key="score_grp_filter")

            show = final_matches if sel_grp == "All Groups" else [m for m in final_matches if m["group"] == sel_grp]
            show = sorted(show, key=lambda x: x["date"], reverse=True)

            for m in show:
                s = m["score_ft"]
                hs, as_ = s[0], s[1]
                t1 = m["team1"]; t2 = m["team2"]
                f1 = TEAMS.get(t1, {}).get("flag", "🏳"); f2 = TEAMS.get(t2, {}).get("flag", "🏳")
                home_w = hs > as_; away_w = as_ > hs; draw = hs == as_

                rc1, rc2, rc3 = st.columns([4, 3, 4])
                with rc1:
                    clr = "#00c07a" if home_w else ("#888" if draw else "#ccc")
                    st.markdown(f"<div style='text-align:right;font-size:1rem;font-weight:600;color:{clr}'>{f1} {t1}</div>", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"<div style='text-align:center;font-size:1.4rem;font-weight:800'>{hs} – {as_}</div>", unsafe_allow_html=True)
                    ht = m.get("score_ht")
                    ht_str = f"HT: {ht[0]}-{ht[1]}" if ht else "FT"
                    st.markdown(f"<div style='text-align:center;color:#888;font-size:0.7rem'>{ht_str} · {m['group'] or m['round']}</div>", unsafe_allow_html=True)
                with rc3:
                    clr = "#00c07a" if away_w else ("#888" if draw else "#ccc")
                    st.markdown(f"<div style='text-align:left;font-size:1rem;font-weight:600;color:{clr}'>{t2} {f2}</div>", unsafe_allow_html=True)

                # Show goalscorers
                g1 = m.get("goals1", [])
                g2 = m.get("goals2", [])
                if g1 or g2:
                    goal_str = ""
                    if g1:
                        goal_str += "⚽ " + ", ".join(f"{g['name']} {g['minute']}\'" for g in g1)
                    if g2:
                        if goal_str: goal_str += "  |  "
                        goal_str += "⚽ " + ", ".join(f"{g['name']} {g['minute']}\'" for g in g2)
                    st.caption(goal_str)

                st.caption(f"📍 {m['venue']}  ·  {m['date']}")
                st.markdown("---")

        # ── UPCOMING ──
        upcoming_known = [m for m in upcoming if m["teams_known"]]
        n_tbd = len(upcoming) - len(upcoming_known)
        if upcoming_known:
            with st.expander(f"📅 Upcoming fixtures ({len(upcoming_known)} with confirmed teams)"):
                if n_tbd:
                    st.caption(f"{n_tbd} later knockout fixture(s) hidden until both teams are determined.")
                upcoming_sorted = sorted(upcoming_known, key=lambda x: x["date"])
                cur_date = None
                for m in upcoming_sorted[:30]:
                    if m["date"] != cur_date:
                        cur_date = m["date"]
                        st.markdown(f"**{m['date']}**")
                    t1 = m["team1"]; t2 = m["team2"]
                    f1 = TEAMS.get(t1, {}).get("flag", "🏳"); f2 = TEAMS.get(t2, {}).get("flag", "🏳")
                    grp = m["group"] or m["round"]
                    st.markdown(f"&nbsp;&nbsp;{f1} {t1} vs {t2} {f2} — {grp} — {m['time']} — {m['venue']}")


# ══════════════════════════════════════
# TAB 5: PREDICTION BACKTEST
# ══════════════════════════════════════
with tabs[4]:
    st.subheader("📸 Prediction Backtest")
    st.caption("Evaluates the model's pre-match predictions against actual results for every completed match")

    all_matches = fetch_wc_matches()
    completed = [m for m in all_matches if m["status"] == "final" and m["score_ft"]
                 and m["team1"] in TEAMS and m["team2"] in TEAMS]

    if not completed:
        st.info("No completed matches yet — check back once the tournament is underway.")
    else:
        # ── Build prediction record for every completed match ──
        records = []
        for m in completed:
            t1, t2 = m["team1"], m["team2"]
            s = m["score_ft"]
            actual_g1, actual_g2 = s[0], s[1]

            if actual_g1 > actual_g2:
                actual_outcome = "team1"
            elif actual_g2 > actual_g1:
                actual_outcome = "team2"
            else:
                actual_outcome = "draw"

            pw1, pd, pw2 = match_probs(t1, t2, group_stage=True)
            xg1, xg2 = expected_goals(t1, t2)

            # Predicted outcome = highest-prob bucket
            if pw1 >= pd and pw1 >= pw2:
                pred_outcome = "team1"
            elif pw2 >= pd and pw2 >= pw1:
                pred_outcome = "team2"
            else:
                pred_outcome = "draw"

            correct = pred_outcome == actual_outcome

            # Predicted winner probability
            if pred_outcome == "team1":
                pred_conf = pw1
            elif pred_outcome == "team2":
                pred_conf = pw2
            else:
                pred_conf = pd

            # Brier score component for this match
            p_actual = pw1 if actual_outcome == "team1" else (pw2 if actual_outcome == "team2" else pd)
            brier = (1 - p_actual) ** 2

            records.append({
                "date":           m["date"],
                "group":          m["group"] or m["round"],
                "team1":          t1,
                "team2":          t2,
                "flag1":          TEAMS[t1]["flag"],
                "flag2":          TEAMS[t2]["flag"],
                "score":          f"{actual_g1}–{actual_g2}",
                "actual":         actual_outcome,
                "pred":           pred_outcome,
                "correct":        correct,
                "pw1":            pw1,
                "pd":             pd,
                "pw2":            pw2,
                "pred_conf":      pred_conf,
                "xg1":            xg1,
                "xg2":            xg2,
                "brier":          brier,
                "p_actual":       p_actual,
            })

        n_total   = len(records)
        n_correct = sum(1 for r in records if r["correct"])
        accuracy  = n_correct / n_total * 100 if n_total else 0
        avg_brier = sum(r["brier"] for r in records) / n_total if n_total else 0
        avg_conf  = sum(r["pred_conf"] for r in records) / n_total * 100 if n_total else 0

        # ── Summary scorecards ──
        # Brier (a proper score) is the primary signal. Top-1 accuracy is shown but
        # de-emphasized: a draw is almost never any match's single most-likely outcome,
        # so an argmax classifier structurally rarely predicts draws — accuracy alone
        # understates a calibrated probabilistic model.
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Avg Brier Score", f"{avg_brier:.3f}",
                   help="Primary metric. (1 − P(actual))². Lower is better. Perfect = 0, random = 0.667.")
        sc2.metric("Matches Evaluated", n_total)
        sc3.metric("Top-1 Accuracy", f"{accuracy:.1f}%", f"{n_correct}/{n_total}",
                   help="Share of matches where the single most-likely outcome was correct. "
                        "Draws are rarely any match's modal outcome, so this caps below 100% by design — "
                        "read it alongside the calibration plot, not on its own.")
        sc4.metric("Avg Predicted Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")

        # ── Accuracy over time chart ──
        records_sorted = sorted(records, key=lambda x: x["date"])
        running_correct = 0
        running_acc = []
        for i, r in enumerate(records_sorted, 1):
            if r["correct"]: running_correct += 1
            running_acc.append(running_correct / i * 100)

        if len(running_acc) > 1:
            fig_acc = go.Figure()
            fig_acc.add_scatter(
                x=list(range(1, len(running_acc)+1)),
                y=running_acc,
                mode="lines+markers",
                line=dict(color="#00c07a", width=2),
                marker=dict(size=7),
                name="Running Accuracy %",
            )
            fig_acc.add_hline(y=50, line_dash="dash", line_color="#555", annotation_text="50% baseline")
            fig_acc.update_layout(
                title="Model Accuracy Over Time (Running %)",
                height=250, margin=dict(t=40, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                yaxis=dict(range=[0, 105], gridcolor="#222", title="Accuracy %"),
                xaxis=dict(gridcolor="#222", title="Match #"),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        # ── Calibration / reliability diagram (pooled one-vs-rest) ──
        # Every match contributes three (predicted prob, did-it-happen) points: one for
        # team1-win, one for draw, one for team2-win. Pooling and binning these shows
        # whether stated probabilities match observed frequencies — the right test for a
        # probabilistic model, and one accuracy can't provide.
        cal_points = []
        for r in records:
            cal_points.append((r["pw1"], 1.0 if r["actual"] == "team1" else 0.0))
            cal_points.append((r["pd"],  1.0 if r["actual"] == "draw"  else 0.0))
            cal_points.append((r["pw2"], 1.0 if r["actual"] == "team2" else 0.0))

        edges = np.linspace(0, 1, 11)
        cal_x, cal_y, cal_n = [], [], []
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            in_bin = [(p, y) for (p, y) in cal_points
                      if (lo <= p < hi) or (i == 9 and p == hi)]
            if in_bin:
                cal_x.append(sum(p for p, _ in in_bin) / len(in_bin) * 100)
                cal_y.append(sum(y for _, y in in_bin) / len(in_bin) * 100)
                cal_n.append(len(in_bin))

        if len(cal_x) >= 2:
            fig_cal = go.Figure()
            fig_cal.add_scatter(
                x=[0, 100], y=[0, 100], mode="lines",
                line=dict(color="#555", dash="dash"), name="Perfect calibration",
                hoverinfo="skip",
            )
            fig_cal.add_scatter(
                x=cal_x, y=cal_y, mode="markers+lines",
                marker=dict(size=[8 + c for c in cal_n], color="#4fc3f7", opacity=0.85,
                            line=dict(color="#0e1a1e", width=1)),
                line=dict(color="#4fc3f7", width=1.5),
                text=[f"{n} predictions in bin" for n in cal_n],
                hovertemplate="Mean predicted: %{x:.1f}%<br>Observed: %{y:.1f}%<br>%{text}<extra></extra>",
                name="Model",
            )
            fig_cal.update_layout(
                title="Calibration / Reliability — pooled one-vs-rest (marker size = sample count)",
                height=300, margin=dict(t=40, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=True,
                legend=dict(orientation="h", y=-0.2),
                xaxis=dict(range=[0, 100], gridcolor="#222", title="Mean predicted probability %"),
                yaxis=dict(range=[0, 100], gridcolor="#222", title="Observed frequency %"),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption("Points on the diagonal are well-calibrated. Below the line = overconfident "
                       "(predicted too high); above = underconfident. Needs a fair sample per bin to read.")

        # ── Confidence vs outcome scatter ──
        if len(records) >= 3:
            fig_conf = go.Figure()
            colors_pts = ["#00c07a" if r["correct"] else "#ff5252" for r in records_sorted]
            fig_conf.add_scatter(
                x=[r["pred_conf"]*100 for r in records_sorted],
                y=[r["p_actual"]*100 for r in records_sorted],
                mode="markers",
                marker=dict(color=colors_pts, size=10, opacity=0.85),
                text=[f"{r['flag1']} {r['team1']} vs {r['flag2']} {r['team2']}<br>{r['score']}" for r in records_sorted],
                hovertemplate="%{text}<br>Pred conf: %{x:.1f}%<br>P(actual): %{y:.1f}%<extra></extra>",
            )
            fig_conf.update_layout(
                title="Predicted Confidence vs P(Actual Outcome) — Green = Correct",
                height=260, margin=dict(t=40, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                xaxis=dict(gridcolor="#222", title="Predicted Confidence %", range=[0, 105]),
                yaxis=dict(gridcolor="#222", title="P(Actual outcome) %", range=[0, 105]),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # ── Per-match detail table ──
        st.markdown("---")
        st.markdown("##### Match-by-Match Prediction Log")

        # Filter controls
        fc1, fc2 = st.columns(2)
        with fc1:
            outcome_filter = st.radio("Show", ["All", "Correct ✅", "Wrong ❌"], horizontal=True, key="bt_outcome_filter")
        with fc2:
            grps_bt = sorted(set(r["group"] for r in records))
            grp_filter = st.selectbox("Group", ["All"] + grps_bt, key="bt_grp_filter")

        filtered = records_sorted
        if outcome_filter == "Correct ✅":
            filtered = [r for r in filtered if r["correct"]]
        elif outcome_filter == "Wrong ❌":
            filtered = [r for r in filtered if not r["correct"]]
        if grp_filter != "All":
            filtered = [r for r in filtered if r["group"] == grp_filter]

        st.caption(f"Showing {len(filtered)} of {n_total} matches")

        for r in filtered:
            # Color coding
            bg = "#0d1f0d" if r["correct"] else "#1f0d0d"
            border = "#00c07a55" if r["correct"] else "#ff525255"
            result_icon = "✅" if r["correct"] else "❌"

            # Outcome label
            def outcome_label(o, t1, t2):
                if o == "team1": return t1
                if o == "team2": return t2
                return "Draw"

            pred_label   = outcome_label(r["pred"],   r["team1"], r["team2"])
            actual_label = outcome_label(r["actual"], r["team1"], r["team2"])

            st.markdown(f"""
            <div style='background:{bg};border:1px solid {border};border-radius:10px;
                        padding:0.75rem 1rem;margin-bottom:0.5rem'>
                <div style='display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap'>
                    <span style='color:#888;font-size:0.72rem;width:5rem'>{r['date']}</span>
                    <span style='font-size:0.72rem;color:#aaa;width:5.5rem'>{r['group']}</span>
                    <span style='font-weight:700;min-width:8rem'>{r['flag1']} {r['team1']}</span>
                    <span style='font-size:1.1rem;font-weight:800;color:#eee;min-width:3.5rem;text-align:center'>{r['score']}</span>
                    <span style='font-weight:700;min-width:8rem'>{r['flag2']} {r['team2']}</span>
                    <span style='flex:1'></span>
                    <span style='font-size:0.75rem;color:#aaa'>
                        Pred: <b style='color:#4fc3f7'>{pred_label}</b> ({r['pred_conf']*100:.0f}%)
                        &nbsp;|&nbsp;
                        Actual: <b style='color:#f0b429'>{actual_label}</b>
                    </span>
                    <span style='font-size:1.2rem'>{result_icon}</span>
                </div>
                <div style='font-size:0.7rem;color:#666;margin-top:0.3rem'>
                    xG: {r['flag1']} {r['xg1']:.2f} – {r['xg2']:.2f} {r['flag2']}
                    &nbsp;|&nbsp; P({r['flag1']}W): {r['pw1']*100:.1f}%
                    · Draw: {r['pd']*100:.1f}%
                    · P({r['flag2']}W): {r['pw2']*100:.1f}%
                    &nbsp;|&nbsp; Brier: {r['brier']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Model notes ──
        st.markdown("---")
        with st.expander("📋 Model Notes"):
            st.markdown("""
            **Data Sources:**
            - FIFA Men's World Rankings (April 1, 2026) — official rankings
            - Elo ratings derived from FIFA ranking points + historical WC performance
            - Attack/defense ratings from confederation-adjusted Elo composites
            - Match results: openfootball open data (github.com/openfootball/world-cup.json)

            **Model Method:**
            - Win probability: Elo-based formula (400-point logistic scale) + attack/defense modifier
            - Draw inflation: Group stage adds ~26% draw probability, scaled down for large Elo gaps
            - Expected goals: Poisson model parameterized by normalized attack vs. opponent defense
            - Scoreline probabilities: exact (product of Poisson marginals), not Monte Carlo
            - Group & tournament sims: matches already played are **locked to their real scores**;
              only remaining fixtures are simulated (50k group / 20k tournament iterations)
            - Knockout bracket: simplified re-drawn knockout (opponents re-drawn each round),
              not a fixed seeded bracket — champion odds are approximate

            **Backtest Metrics:**
            - **Avg Brier** (primary): (1 − P(actual outcome))² — proper score; lower is better; random = 0.667, perfect = 0
            - **Calibration plot** (primary): pooled one-vs-rest predicted prob vs observed frequency
            - **Top-1 accuracy** (secondary): a draw is rarely any match's single most-likely outcome,
              so an argmax classifier predicts draws only in genuine toss-ups — accuracy alone
              understates a well-calibrated model and should be read with the calibration plot
            - Predictions use pre-tournament ratings only (no in-tournament form updates)

            **Limitations:**
            - Injury/suspension impact not modeled
            - In-tournament form / rating updates not incorporated
            - Knockout bracket seeding simplified (re-drawn each round)
            """)

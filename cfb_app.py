"""
CFB Weekly Slate — Win Probability & Spreads
--------------------------------------------
College football sibling of the MLB Daily Slate app. Same philosophy:
frozen model, hardcoded price gate, full-slate pick tracker with
auto-grading, GitHub-backed persistence.

Install dependencies:
    pip install streamlit pandas numpy requests

Run:
    streamlit run cfb_app.py

Data source: ESPN's public college football API — no API key needed.
Team stats (PPG for/against, margin, win%) are computed from actual game
scores pulled week by week, so the whole app is keyless like the MLB one.
"""

import base64
import datetime
import json
import math
import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from zoneinfo import ZoneInfo

st.set_page_config(page_title="CFB Weekly Slate", page_icon="🏈", layout="wide")

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


# ── App clock ──────────────────────────────────────────────────────────────────
# Same lesson as the MLB app: Streamlit Cloud runs on UTC, so naive
# datetime.today() rolls to "tomorrow" at 8 PM Eastern. Every date in this
# app uses the Eastern-time football day via these helpers. Never call
# datetime.datetime.today()/now() directly anywhere below.
APP_TZ = ZoneInfo("America/New_York")


def now_et() -> datetime.datetime:
    return datetime.datetime.now(APP_TZ)


def today_et(fmt: str = "%Y-%m-%d") -> str:
    return now_et().strftime(fmt)


# CFB season year = the calendar year of the fall the season starts in.
# (Bowls in January still belong to the previous year's season; ESPN's
# API uses the same convention, so this stays correct through January.)
_now = now_et()
SEASON = _now.year if _now.month >= 6 else _now.year - 1
PRIOR_SEASON = SEASON - 1

# ── MODEL FREEZE ───────────────────────────────────────────────────────────────
# Same discipline as the MLB app: this version is FROZEN for forward
# measurement. Every logged pick is tagged with this string. No
# parameter/formula changes until the tracker holds ≥100 graded picks
# PER MARKET (ML and ATS separately) for this version.
MODEL_VERSION = "cfb-v1.1-frozen-2026-08-27"

# ── GATE THRESHOLD — deliberately NOT adjustable in the UI ─────────────────────
# Carried over verbatim from the MLB app at Juan's request: no in-app knob,
# so in-the-moment eagerness can't loosen the discipline. Changing this
# requires editing code — that friction is the feature. Same 3.0pp cushion
# applies to BOTH markets: for ML it means model win prob must clear the
# price's break-even by 3pp; for ATS it means model cover prob must clear
# the juice's break-even (52.4% at -110) by 3pp — which works out to the
# model's line disagreeing with the market by roughly 2+ points.
GATE_THRESH_PP = 3.0

# ── Margin model constants ─────────────────────────────────────────────────────
# HFA: college home-field advantage has measured ~2.5 points in recent
# seasons (down from the historical ~3–3.5). Anchored to the public base
# rate, not tuned to results. ESPN flags neutral-site games; those get 0.
HFA_POINTS = 2.5

# K_MARGIN: converts the composite z-score gap into points. FBS team
# quality spans roughly ±2.5σ on the composite, and the widest real
# point spreads between FBS teams run into the 40s. 13.0 points per σ
# puts a 2.5σ-vs-−2.5σ mismatch at ~65 raw points before clipping and a
# typical 1σ edge at ~13 — consistent with how books space FBS lines.
# Reasoned default, not a fitted value.
K_MARGIN = 13.0

# SIGMA_WIN: std dev of actual margin around the true expected margin,
# used to convert projected margin → win probability via the normal CDF.
# CFB final margins scatter around closing spreads with σ ≈ 15.5–16.5
# (noisier than the NFL's ~13.5). 16.0 is the round anchor.
SIGMA_WIN = 16.0

# SIGMA_COVER: same distribution drives cover probability — the chance the
# actual margin lands past the market line given our projected margin.
# Kept as its own named constant (15.5) because cover math is the part
# most worth revisiting once the ATS tracker has data: if cushion buckets
# show overconfidence, THIS is the number that's too small.
SIGMA_COVER = 15.5


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── ESPN endpoints ─────────────────────────────────────────────────────────────
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
FBS_GROUP = "80"   # ESPN group id for FBS


def _espn_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=86400)
def fetch_fbs_teams() -> dict:
    """{team_id(str): display_name} for all FBS teams. Cached a day."""
    out = {}
    j = _espn_get(f"{ESPN}/teams", {"limit": 400, "groups": FBS_GROUP})
    for sport in j.get("sports", []):
        for lg in sport.get("leagues", []):
            for t in lg.get("teams", []):
                tm = t.get("team", {})
                if tm.get("id"):
                    out[str(tm["id"])] = tm.get("displayName", tm.get("name", ""))
    return out


def _parse_event(ev: dict) -> dict | None:
    """Flatten one ESPN scoreboard event into the fields we use."""
    try:
        comp = ev["competitions"][0]
        home = away = None
        for c in comp.get("competitors", []):
            side = {"id": str(c["team"]["id"]),
                    "name": c["team"].get("displayName", ""),
                    "score": int(c.get("score") or 0),
                    "record": ""}
            for rec in c.get("records", []):
                if rec.get("type") == "total" or rec.get("name") == "overall":
                    side["record"] = rec.get("summary", "")
            if c.get("homeAway") == "home":
                home = side
            else:
                away = side
        if not home or not away:
            return None
        status = comp.get("status", {}).get("type", {})
        # Market odds, when ESPN carries them: 'details' like "UGA -6.5",
        # spread as HOME-relative number, overUnder. Used only to PREFILL
        # the gate form — the model never reads the market.
        odds = {}
        for o in comp.get("odds", []) or []:
            if "spread" in o or "overUnder" in o:
                odds = {"home_spread": o.get("spread"),
                        "over_under": o.get("overUnder"),
                        "details": o.get("details", "")}
                break
        date_utc = ev.get("date", "")
        try:
            dt = datetime.datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
            dt_et = dt.astimezone(APP_TZ)
            kick_label = dt_et.strftime("%a %I:%M %p ET").replace(" 0", " ")
            kick_date = dt_et.strftime("%Y-%m-%d")
        except Exception:
            kick_label, kick_date = "", ""
        return {
            "home_id": home["id"], "home": home["name"],
            "home_score": home["score"], "home_record": home["record"],
            "away_id": away["id"], "away": away["name"],
            "away_score": away["score"], "away_record": away["record"],
            "neutral": bool(comp.get("neutralSite", False)),
            "completed": bool(status.get("completed", False)),
            "state": status.get("state", ""),          # pre / in / post
            "status_detail": status.get("shortDetail", ""),
            "kick_label": kick_label, "kick_date": kick_date,
            "venue": (comp.get("venue", {}) or {}).get("fullName", ""),
            "odds": odds,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_week_slate(season: int, week: int) -> list:
    """All FBS scoreboard events for one week of the regular season."""
    j = _espn_get(f"{ESPN}/scoreboard",
                  {"groups": FBS_GROUP, "limit": 300,
                   "dates": season, "seasontype": 2, "week": week})
    games = [_parse_event(ev) for ev in j.get("events", [])]
    return [g for g in games if g]


@st.cache_data(ttl=1800)
def fetch_current_week(season: int) -> int:
    """ESPN's notion of the current week (falls back to 1)."""
    try:
        j = _espn_get(f"{ESPN}/scoreboard", {"groups": FBS_GROUP, "limit": 1})
        wk = j.get("week", {}).get("number")
        yr = j.get("season", {}).get("year")
        if wk and yr == season:
            return int(wk)
    except Exception:
        pass
    return 1


@st.cache_data(ttl=600)
def fetch_results_for_date(date_iso: str) -> list:
    """Final scores for all FBS games on a date (for auto-grading)."""
    ymd = date_iso.replace("-", "")
    j = _espn_get(f"{ESPN}/scoreboard",
                  {"groups": FBS_GROUP, "limit": 300, "dates": ymd})
    games = [_parse_event(ev) for ev in j.get("events", [])]
    return [g for g in games if g]


# ── Team season stats, computed from game scores ───────────────────────────────
# There is no keyless statsapi for CFB, so the stat layer is built the way
# simple rating systems are: pull every completed game's score and derive
# PPG for/against, margin/game, and win% per team. Games against non-FBS
# opponents are EXCLUDED from the stat lines — beating an FCS team 56-3
# says almost nothing and would inflate early-season offense numbers.
# Known, accepted limitation (documented, not hidden): these stats are NOT
# opponent-adjusted. Early in the season a soft schedule inflates a team.
# The confidence tiers flag low game counts for exactly this reason.

@st.cache_data(ttl=1800)
def fetch_season_team_stats(season: int, upto_week: int | None = None) -> dict:
    """
    {team_id: {pf_pg, pa_pg, mpg, wpct, wins, losses, games}}
    from completed FBS-vs-FBS games. upto_week limits how deep to scan
    (weeks 1..upto_week); None scans the full regular season (1..16).
    Cached 30 min; the prior season's call is effectively static.
    """
    fbs = fetch_fbs_teams()
    acc = {}     # id -> [pf, pa, w, l, g]
    gamelog = {}  # id -> [(opp_id, pf, pa)] — feeds the opponent adjustment

    def bump(tid, opp, pf, pa, won):
        a = acc.setdefault(tid, [0, 0, 0, 0, 0])
        a[0] += pf; a[1] += pa
        a[2] += 1 if won else 0
        a[3] += 0 if won else 1
        a[4] += 1
        gamelog.setdefault(tid, []).append((opp, pf, pa))

    last_week = upto_week or 16
    for wk in range(1, last_week + 1):
        try:
            slate = fetch_week_slate(season, wk)
        except Exception:
            continue
        for g in slate:
            if not g["completed"]:
                continue
            # FBS-vs-FBS only — see comment above.
            if g["home_id"] not in fbs or g["away_id"] not in fbs:
                continue
            hs, as_ = g["home_score"], g["away_score"]
            bump(g["home_id"], g["away_id"], hs, as_, hs > as_)
            bump(g["away_id"], g["home_id"], as_, hs, as_ > hs)

    # Keyed by ESPN team ID — NOT display name. Names can differ between
    # ESPN's /teams and /scoreboard endpoints for the same school, and a
    # name-keyed join silently drops the team (which once flagged an
    # FBS-vs-FBS game as "non-FBS"). IDs are identical across endpoints.
    out = {}
    for tid, (pf, pa, w, l, gp) in acc.items():
        if gp == 0:
            continue
        out[tid] = {
            "pf_pg": pf / gp, "pa_pg": pa / gp,
            "mpg": (pf - pa) / gp,
            "wpct": w / (w + l) * 100 if (w + l) else 50.0,
            "wins": w, "losses": l, "games": gp,
        }
    return opponent_adjust(out, gamelog)


def opponent_adjust(stats: dict, gamelog: dict, n_iter: int = 30,
                    damp: float = 0.7) -> dict:
    """
    SRS-style iterative opponent adjustment — the answer to "PPG doesn't
    consider level of competition." Uses ONLY the game scores already
    pulled; no new data source.

    Idea: a point scored against a stingy defense is worth more than one
    scored against a sieve. Each pass re-values every team's offense and
    defense against its opponents' CURRENT adjusted ratings:

        adj_off(i) = mean over i's games of (pts scored − adj_def(opp))
                     + league mean
        adj_def(i) = mean over i's games of (pts allowed − adj_off(opp))
                     + league mean          (lower = better defense)

    Iterated to convergence (damped, so early-season sparse schedules —
    where the opponent graph is barely connected — can't oscillate), then
    adjusted values are clamped to a sane band. wpct stays RAW: a record
    is a record; the schedule context lives in the adjusted margin, which
    carries 75% of the record composite anyway. mpg is recomputed from
    the adjusted scoring lines, so K_MARGIN and the z-score anchors keep
    their meaning (the adjustment preserves the league mean).
    """
    if not stats:
        return stats
    mean_off = sum(s["pf_pg"] for s in stats.values()) / len(stats)
    mean_def = sum(s["pa_pg"] for s in stats.values()) / len(stats)
    adj_off = {t: s["pf_pg"] for t, s in stats.items()}
    adj_def = {t: s["pa_pg"] for t, s in stats.items()}
    for _ in range(n_iter):
        new_off, new_def = {}, {}
        for t, s in stats.items():
            games = gamelog.get(t, [])
            if not games:
                new_off[t], new_def[t] = adj_off[t], adj_def[t]
                continue
            # Opponents missing from stats (shouldn't happen for FBS-vs-FBS
            # rows, but be safe) count as league average.
            o = sum(pf - adj_def.get(opp, mean_def)
                    for opp, pf, pa in games) / len(games) + mean_def
            d = sum(pa - adj_off.get(opp, mean_off)
                    for opp, pf, pa in games) / len(games) + mean_off
            new_off[t] = damp * o + (1 - damp) * adj_off[t]
            new_def[t] = damp * d + (1 - damp) * adj_def[t]
        # Re-center every pass: the system has a free "gauge" mode (add a
        # constant to all defenses, subtract it from all offenses — every
        # game prediction is unchanged but the levels drift). Pinning the
        # means to the raw league means each iteration kills the drift.
        off_shift = mean_off - sum(new_off.values()) / len(new_off)
        def_shift = mean_def - sum(new_def.values()) / len(new_def)
        adj_off = {t: v + off_shift for t, v in new_off.items()}
        adj_def = {t: v + def_shift for t, v in new_def.items()}
    # Clamp to a sane band: 3σ of the league spread around the mean.
    lo_o, hi_o = mean_off - 21, mean_off + 21
    lo_d, hi_d = mean_def - 21, mean_def + 21
    for t, s in stats.items():
        s["raw_pf_pg"], s["raw_pa_pg"] = s["pf_pg"], s["pa_pg"]
        s["pf_pg"] = min(hi_o, max(lo_o, adj_off[t]))
        s["pa_pg"] = min(hi_d, max(lo_d, adj_def[t]))
        s["mpg"] = s["pf_pg"] - s["pa_pg"]
        s["sos"] = round((s["pf_pg"] - s["raw_pf_pg"])
                         + (s["raw_pa_pg"] - s["pa_pg"]), 1)
    return stats


# League anchors for z-scoring. FBS scoring runs ~28-29 PPG with a ~7 point
# std across teams; margin/game std ~9; win% std ~22pp (CFB records spread
# far wider than MLB's because talent gaps are wider). Fixed anchors (like
# the MLB app's LEAGUE_AVG table) so a drifting league mean can't silently
# rescale every rating mid-season.
LEAGUE_AVG = {"pf_pg": 28.5, "pa_pg": 28.5, "mpg": 0.0, "wpct": 50.0}
LEAGUE_STD = {"pf_pg": 7.0,  "pa_pg": 7.0,  "mpg": 9.0, "wpct": 22.0}


def zscore(val: float, key: str, inv: bool = False) -> float:
    z = (val - LEAGUE_AVG[key]) / LEAGUE_STD[key]
    z = max(-2.5, min(2.5, z))
    return -z if inv else z


def blend_prior(cur: dict | None, prior: dict | None, fade_games: int) -> tuple[dict | None, float]:
    """
    The CFB answer to the MLB app's season/recent blend — but pointed
    backward: at 0 current-season games the model runs 100% on LAST
    season's numbers, fading linearly to 100% current by `fade_games`.
    This is what makes the app usable in Week 1, when current-season
    stats don't exist yet. Returns (blended_stats, prior_weight).
    Roster turnover makes prior-season data imperfect — that's why the
    fade is fast (default 6 games) and why early-season confidence is
    capped in calc_confidence.
    """
    if cur is None and prior is None:
        return None, 0.0
    if cur is None:
        return dict(prior), 1.0
    gp = cur.get("games", 0)
    w_prior = max(0.0, (fade_games - gp) / fade_games)
    if prior is None or w_prior == 0:
        return dict(cur), 0.0
    keys = ["pf_pg", "pa_pg", "mpg", "wpct"]
    blended = {k: cur[k] * (1 - w_prior) + prior[k] * w_prior for k in keys}
    blended["sos"] = round(cur.get("sos", 0) * (1 - w_prior)
                           + prior.get("sos", 0) * w_prior, 1)
    blended["games"] = gp
    blended["wins"], blended["losses"] = cur.get("wins", 0), cur.get("losses", 0)
    return blended, w_prior


def build_components(s: dict) -> tuple[float, float, float]:
    """(offense, defense, record) component z-scores for one team.
    Record composite mirrors the MLB app's W% 25 / RD 75 split: margin
    per game is the far less luck-contaminated of the two signals (CFB
    close-game records are notoriously random), so it carries the weight."""
    off = zscore(s["pf_pg"], "pf_pg")
    dfn = zscore(s["pa_pg"], "pa_pg", inv=True)
    rec = 0.25 * zscore(s["wpct"], "wpct") + 0.75 * zscore(s["mpg"], "mpg")
    return off, dfn, rec


def calc_margin(sh: dict, sa: dict, neutral: bool,
                w_off: float, w_def: float, w_rec: float) -> float:
    """Projected HOME margin in points (positive = home wins by that)."""
    off_h, def_h, rec_h = build_components(sh)
    off_a, def_a, rec_a = build_components(sa)
    rate_h = off_h * w_off + def_h * w_def + rec_h * w_rec
    rate_a = off_a * w_off + def_a * w_def + rec_a * w_rec
    margin = (rate_h - rate_a) * K_MARGIN
    if not neutral:
        margin += HFA_POINTS
    return margin


def margin_to_probs(margin: float) -> tuple[float, float]:
    """Projected home margin → (home win prob, away win prob).
    Same structural principle as the MLB app's logistic fix: probability
    depends only on the GAP, via the margin distribution — not on any
    absolute level that could drift."""
    p_home = norm_cdf(margin / SIGMA_WIN)
    return p_home, 1.0 - p_home


def cover_prob_home(pred_margin: float, home_spread: float) -> float:
    """P(home covers) given our projected margin and the market's
    home-relative spread (negative = home favored, e.g. -6.5 means home
    must win by 7+). Home covers when actual margin > -home_spread."""
    return norm_cdf((pred_margin + home_spread) / SIGMA_COVER)


def project_score(sh: dict, sa: dict, neutral: bool) -> tuple[float, float, float]:
    """Projected (home_pts, away_pts, total). Each side's expectation is
    the average of its offense PPG and the opponent's PPG allowed —
    the standard first-order matchup estimate."""
    ph = (sh["pf_pg"] + sa["pa_pg"]) / 2.0
    pa = (sa["pf_pg"] + sh["pa_pg"]) / 2.0
    if not neutral:
        ph += HFA_POINTS / 2.0
        pa -= HFA_POINTS / 2.0
    ph, pa = max(3.0, ph), max(3.0, pa)
    return round(ph), round(pa), round(ph + pa)


def breakeven_prob(odds: float) -> float:
    """Break-even win probability implied by an American price (includes vig)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def unit_profit(odds: float) -> float:
    """Profit in units on a 1u winning bet at an American price."""
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def matchup_key(g: dict) -> str:
    """Unique, human-readable identity for a game. No doubleheaders in
    CFB, so away @ home is already unique within a date."""
    return f"{g['away']} @ {g['home']}"


def fmt_spread(x: float) -> str:
    """Display a home-relative spread the way books quote it."""
    if abs(x) < 0.25:
        return "PK"
    return f"{x:+g}"


def calc_confidence(g: dict, sh: dict, sa: dict, w_prior: float,
                    pct_h: float, pct_a: float, pred_margin: float,
                    fbs_both: bool) -> tuple[str, str, str, list]:
    """Mirrors the MLB tiers: High / Moderate / Low / Conflicted, with
    plain-language reasons. CFB-specific signals replace the SP logic:
    prior-season reliance, low game counts, and FCS opponents cap the tier.
    Thresholds: prob gaps map back through SIGMA_WIN — strong ≥35pp is a
    ~7.5+ point projected edge, moderate ≥15pp is ~3+."""
    prob_winner = g["home"] if pct_h >= pct_a else g["away"]
    margin_winner = g["home"] if pred_margin >= 0 else g["away"]
    models_agree = prob_winner == margin_winner   # structurally always true
    prob_gap = abs(pct_h - pct_a)
    prob_strength = ("strong" if prob_gap >= 35
                     else "moderate" if prob_gap >= 15 else "narrow")

    # Scoring leader vs record leader — the CFB analog of the MLB app's
    # OPS-vs-record split: schedule strength is usually the culprit.
    score_leader = g["home"] if (sh["pf_pg"] - sh["pa_pg"]) > (sa["pf_pg"] - sa["pa_pg"]) else g["away"]
    rec_leader = g["home"] if sh["wpct"] > sa["wpct"] else g["away"]
    split = score_leader != rec_leader

    low_data = min(sh.get("games", 0), sa.get("games", 0)) < 3
    prior_heavy = w_prior >= 0.5

    if not fbs_both:
        level, emoji, color = "Low", "🟠", "#f5a623"
    elif prob_strength == "strong" and not split and not prior_heavy and not low_data:
        level, emoji, color = "High", "🟢", "#00c07a"
    elif prob_strength in ("strong", "moderate"):
        level, emoji, color = "Moderate", "🟡", "#f5c842"
    else:
        level, emoji, color = "Low", "🟠", "#f5a623"

    reasons = []
    if not fbs_both:
        reasons.append("One side is a **non-FBS opponent** — no comparable "
                       "stat line exists, so the model output here is a "
                       "placeholder, not a read. Treat as no-play.")
    if prior_heavy:
        reasons.append(f"Model is running **{w_prior:.0%} on last season's "
                       f"data** — rosters turn over hard in CFB, so early-"
                       f"season projections carry extra uncertainty by design.")
    elif low_data:
        reasons.append("Fewer than 3 current-season games for at least one "
                       "team — stats are small-sample and unadjusted for "
                       "schedule.")
    if split:
        reasons.append(f"**{score_leader}** has the better scoring margin but "
                       f"**{rec_leader}** has the better record — schedule "
                       f"difficulty is probably the difference.")
    if prob_strength == "narrow":
        reasons.append(f"The edge is slim ({prob_gap:.0f}pp, ~"
                       f"{abs(pred_margin):.0f} pts) — inside one score, "
                       f"where CFB variance dominates.")
    if not reasons:
        reasons.append(f"Offense, defense, and record all point the same way "
                       f"for **{prob_winner}** with a {prob_strength} edge.")
    return level, emoji, color, reasons


# ── GitHub-backed storage (for Streamlit Cloud hosting) ───────────────────────
# Identical mechanism to the MLB app — see that file's long comment for the
# full rationale. Short version: Streamlit Cloud wipes local files on every
# reboot, so with [github] secrets configured the log lives in the repo on a
# separate "data" branch (writes to the watched branch would redeploy the
# app on every save). The log FILENAME differs from the MLB app's
# (cfb_pick_log.json) so both apps can share one repo/branch without
# clobbering each other.
#
# Setup (once):
#   1. GitHub → Settings → Developer settings → Fine-grained tokens → new
#      token. Repository access: ONLY this repo. Permissions: Contents,
#      Read and write.
#   2. Streamlit Cloud → app → Settings → Secrets:
#         [github]
#         token  = "github_pat_..."
#         repo   = "your-username/your-repo"
#         branch = "data"
#   3. Never commit the token to the repo.

GH_API = "https://api.github.com"
PICK_LOG_FILE = "cfb_pick_log.json"
PICK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             PICK_LOG_FILE)


def _gh_cfg():
    try:
        gh = st.secrets["github"]
        # .strip() everywhere: stray whitespace from copy-paste = 401.
        return {"token": str(gh["token"]).strip(),
                "repo": str(gh["repo"]).strip().strip("/"),
                "branch": str(gh.get("branch", "data")).strip()}
    except Exception:
        return None


def _gh_error_hint(err) -> str:
    s = str(err)
    if "401" in s:
        return ("→ GitHub rejected the token (bad credentials). Re-copy the "
                "FULL token into Streamlit Secrets (watch for truncation or "
                "stray spaces), and confirm it hasn't expired.")
    if "404" in s:
        return ("→ Repo not reachable with this token. Check 'owner/name' "
                "spelling and that the token's access includes this repo.")
    if "403" in s:
        return ("→ Access refused. Check the token has Contents: Read and "
                "write permission on this repo.")
    return ""


def _gh_headers(cfg):
    return {"Authorization": f"Bearer {cfg['token']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _gh_ensure_branch(cfg):
    flag = f"_gh_branch_ok_{cfg['branch']}"
    if st.session_state.get(flag):
        return True
    try:
        r = requests.get(f"{GH_API}/repos/{cfg['repo']}/branches/{cfg['branch']}",
                         headers=_gh_headers(cfg), timeout=15)
        if r.status_code == 200:
            st.session_state[flag] = True
            return True
        repo = requests.get(f"{GH_API}/repos/{cfg['repo']}",
                            headers=_gh_headers(cfg), timeout=15).json()
        base = repo.get("default_branch", "main")
        sha = requests.get(f"{GH_API}/repos/{cfg['repo']}/branches/{base}",
                           headers=_gh_headers(cfg), timeout=15
                           ).json()["commit"]["sha"]
        c = requests.post(f"{GH_API}/repos/{cfg['repo']}/git/refs",
                          headers=_gh_headers(cfg),
                          json={"ref": f"refs/heads/{cfg['branch']}", "sha": sha},
                          timeout=15)
        ok = c.status_code in (200, 201, 422)   # 422 = already exists (race)
        st.session_state[flag] = ok
        return ok
    except Exception as e:
        st.error(f"GitHub branch check failed: {e}")
        return False


def _gh_get_file(cfg, path):
    r = requests.get(f"{GH_API}/repos/{cfg['repo']}/contents/{path}",
                     params={"ref": cfg["branch"]},
                     headers=_gh_headers(cfg), timeout=15)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    j = r.json()
    raw = base64.b64decode(j["content"]).decode("utf-8")
    return json.loads(raw), j["sha"]


def _gh_put_file(cfg, path, obj, sha, message):
    body = {"message": message, "branch": cfg["branch"],
            "content": base64.b64encode(
                json.dumps(obj, indent=1).encode("utf-8")).decode("ascii")}
    if sha:
        body["sha"] = sha
    r = requests.put(f"{GH_API}/repos/{cfg['repo']}/contents/{path}",
                     headers=_gh_headers(cfg), json=body, timeout=15)
    if r.status_code in (409, 422):
        _, fresh = _gh_get_file(cfg, path)
        if fresh:
            body["sha"] = fresh
        elif "sha" in body:
            del body["sha"]
        r = requests.put(f"{GH_API}/repos/{cfg['repo']}/contents/{path}",
                         headers=_gh_headers(cfg), json=body, timeout=15)
    r.raise_for_status()
    return r.json()["content"]["sha"]


def load_pick_log() -> list:
    cfg = _gh_cfg()
    if cfg is None:
        if not os.path.exists(PICK_LOG_PATH):
            return []
        try:
            with open(PICK_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    if "_pick_log_cache" in st.session_state:
        return st.session_state["_pick_log_cache"]
    try:
        data, sha = _gh_get_file(cfg, PICK_LOG_FILE)
    except Exception as e:
        st.error(f"Couldn't load pick log from GitHub ({e}). "
                 f"{_gh_error_hint(e)} Showing empty log — do NOT save "
                 f"grades until this resolves.")
        return []
    st.session_state["_pick_log_cache"] = data or []
    st.session_state["_pick_log_sha"] = sha
    return st.session_state["_pick_log_cache"]


def save_pick_log(rows: list):
    try:
        with open(PICK_LOG_PATH, "w") as f:
            json.dump(rows, f, indent=1)
    except Exception:
        pass
    cfg = _gh_cfg()
    if cfg is None:
        return
    if not _gh_ensure_branch(cfg):
        st.error("Pick log NOT saved to GitHub (branch unavailable).")
        return
    try:
        new_sha = _gh_put_file(cfg, PICK_LOG_FILE, rows,
                               st.session_state.get("_pick_log_sha"),
                               "tracker: update cfb pick log")
        st.session_state["_pick_log_cache"] = rows
        st.session_state["_pick_log_sha"] = new_sha
    except Exception as e:
        st.error(f"Pick log NOT saved to GitHub: {e} {_gh_error_hint(e)}")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏈 CFB Weekly Slate")
    st.caption(f"Season: {SEASON} · ESPN data · refreshes every 30 min")

    st.markdown("---")
    st.markdown("##### Settings")
    fade_games = st.select_slider(
        "Prior-season fade (games to fully current)",
        options=[4, 6, 8], value=6,
        help=f"At 0 games played the model runs 100% on {PRIOR_SEASON} "
             f"stats, fading linearly to 100% {SEASON} stats by this many "
             f"games. Faster fade = trusts the new roster sooner.")

    st.markdown("---")
    st.markdown("##### Model weights")
    # Defaults mirror the MLB app's 40/40/20 reasoning: record is the most
    # downstream, most luck-contaminated signal (CFB close-game outcomes
    # are close to coin flips) and is largely redundant with the scoring
    # numbers that produce it. Sliders stay live.
    w_off_raw = st.slider("Offense weight", 0, 100, 40, step=5)
    w_def_raw = st.slider("Defense weight", 0, 100, 40, step=5)
    w_rec_raw = st.slider("Record weight",  0, 100, 20, step=5)
    total_w = (w_off_raw + w_def_raw + w_rec_raw) or 1
    w_off = w_off_raw / total_w
    w_def = w_def_raw / total_w
    w_rec = w_rec_raw / total_w

    st.markdown("---")
    cur_week_default = fetch_current_week(SEASON)
    week = st.number_input("Week", min_value=1, max_value=16,
                           value=int(cur_week_default), step=1)

    with st.spinner(f"Loading {SEASON} results..."):
        cur_stats = fetch_season_team_stats(SEASON, upto_week=int(week))
    with st.spinner(f"Loading {PRIOR_SEASON} baselines..."):
        prior_stats = fetch_season_team_stats(PRIOR_SEASON)
    fbs_teams = fetch_fbs_teams()   # {id: display_name}

    n_cur = len(cur_stats)
    if n_cur == 0:
        st.info(f"📊 No {SEASON} games completed yet — running fully on "
                f"{PRIOR_SEASON} data ({len(prior_stats)} teams). This is "
                f"the designed Week-1 behavior, not an error.")
    else:
        gp_list = [v["games"] for v in cur_stats.values()]
        avg_gp = sum(gp_list) / len(gp_list)
        st.success(f"✅ {SEASON} stats: {n_cur} teams · "
                   f"avg {avg_gp:.1f} games")
        if avg_gp < 3:
            st.warning("⚠️ Very early season — projections lean on "
                       f"{PRIOR_SEASON} data and small samples.", icon="🏈")
    st.caption(f"{PRIOR_SEASON} baselines: {len(prior_stats)} teams · "
               f"FBS teams tracked: {len(fbs_teams)}")


# ── Main view ──────────────────────────────────────────────────────────────────
(tab_week,) = st.tabs(["🏈 This Week's Games"])

with tab_week:
    st.header(f"Week {int(week)} Slate — {SEASON}")

    with st.spinner("Loading slate..."):
        try:
            slate = fetch_week_slate(SEASON, int(week))
        except Exception as e:
            st.error(f"Couldn't load the slate from ESPN: {e}")
            slate = []

    if not slate:
        st.info("No games found for this week yet.")
        st.stop()

    # Generic placeholder rating for non-FBS opponents so the card can
    # still render — flagged hard by calc_confidence as a no-play. z ≈ -2.5
    # on everything: roughly "worst FBS team" as a floor.
    FCS_PLACEHOLDER = {"pf_pg": 17.0, "pa_pg": 38.0, "mpg": -21.0,
                       "wpct": 25.0, "wins": 0, "losses": 0, "games": 0}

    slate_results = []
    for g in slate:
        # Membership and stat joins are ALL by ESPN team ID (see the
        # comment in fetch_season_team_stats): display names are not a
        # reliable join key across ESPN endpoints.
        h_fbs = g["home_id"] in fbs_teams
        a_fbs = g["away_id"] in fbs_teams
        fbs_both = h_fbs and a_fbs

        sh_blend, wp_h = blend_prior(cur_stats.get(g["home_id"]),
                                     prior_stats.get(g["home_id"]), fade_games) \
            if h_fbs else (dict(FCS_PLACEHOLDER), 0.0)
        sa_blend, wp_a = blend_prior(cur_stats.get(g["away_id"]),
                                     prior_stats.get(g["away_id"]), fade_games) \
            if a_fbs else (dict(FCS_PLACEHOLDER), 0.0)
        # FBS teams with no games in either season's data (newly promoted,
        # or a data gap) also get the floor rating — but flagged as a DATA
        # gap, not mislabeled "non-FBS".
        data_gap = False
        if sh_blend is None:
            sh_blend, fbs_both, data_gap = dict(FCS_PLACEHOLDER), False, True
        if sa_blend is None:
            sa_blend, fbs_both, data_gap = dict(FCS_PLACEHOLDER), False, True
        w_prior = max(wp_h, wp_a)

        pred_margin = calc_margin(sh_blend, sa_blend, g["neutral"],
                                  w_off, w_def, w_rec)
        p_h, p_a = margin_to_probs(pred_margin)
        pct_h, pct_a = round(p_h * 100, 1), round(p_a * 100, 1)
        proj_h, proj_a, proj_total = project_score(sh_blend, sa_blend,
                                                   g["neutral"])
        pick = g["home"] if pct_h >= pct_a else g["away"]
        # Model line, quoted book-style for the HOME team: projected home
        # margin of +7.5 → model line "home -7.5". Snapped to 0.5.
        model_line_home = -round(pred_margin * 2) / 2

        level, emoji, color, reasons = calc_confidence(
            g, sh_blend, sa_blend, w_prior, pct_h, pct_a,
            pred_margin, fbs_both)
        if data_gap:
            reasons[0] = ("An **FBS team here has no games in either "
                          "season's data** (newly promoted program or a "
                          "data gap) — a placeholder rating is in use. "
                          "Treat as no-play.")

        slate_results.append({
            **g,
            "sh": sh_blend, "sa": sa_blend, "w_prior": w_prior,
            "fbs_both": fbs_both,
            "pred_margin": pred_margin,
            "model_line_home": model_line_home,
            "home_pct": pct_h, "away_pct": pct_a,
            "proj_home": proj_h, "proj_away": proj_a, "proj_total": proj_total,
            "prob_pick": pick,
            "conf_level": level, "conf_emoji": emoji,
            "conf_color": color, "conf_reasons": reasons,
        })

    # ── Price gate — all price entry in ONE place, one Enter to apply ─────
    # Same design as the MLB app (see its comments), extended to two
    # markets per game at Juan's request:
    #   ML  — price on the model's moneyline pick → gate on win prob.
    #   ATS — the market's HOME spread + the juice on the model's side of
    #         that line → gate on COVER prob. The model's side of the
    #         market line is whichever side its projected margin says
    #         covers; that can differ from its ML pick (e.g. model likes
    #         the favorite to win but the dog to cover).
    # The cushion threshold is fixed at GATE_THRESH_PP for both markets —
    # no UI control, by design.
    upcoming = [g for g in slate_results if not g["completed"]]
    gate_ml = {}
    gate_ats = {}
    if upcoming:
        st.subheader("💰 Price gate")
        with st.form("odds_form"):
            st.caption("For each game: the market's HOME spread (book "
                       "convention, negative = home favored — prefilled "
                       "from ESPN when available), the juice on the "
                       "model's side of that spread, and the ML price on "
                       "the model's pick. Enter them all, then Apply once. "
                       "Leave a field at 0 to skip that market.")
            for g in upcoming:
                mkey = matchup_key(g)
                pick_p = (g["home_pct"] if g["prob_pick"] == g["home"]
                          else g["away_pct"])
                fc0, fc1, fc2, fc3 = st.columns([2.4, 0.9, 0.9, 0.9])
                with fc0:
                    st.markdown(
                        f"<div style='padding-top:8px;font-size:13px;'>"
                        f"{mkey}<br><span style='color:#888;'>ML pick: "
                        f"<b>{g['prob_pick']}</b> ({pick_p:.1f}%) · model "
                        f"line: <b>{g['home']} "
                        f"{fmt_spread(g['model_line_home'])}</b></span></div>",
                        unsafe_allow_html=True)
                # Prefill the market spread from ESPN odds when carried.
                espn_sp = g["odds"].get("home_spread") if g.get("odds") else None
                default_sp = float(espn_sp) if espn_sp is not None else 0.0
                with fc1:
                    st.number_input("home spread", value=default_sp, step=0.5,
                                    key=f"sp_{mkey}", format="%.1f",
                                    help="Market spread for the HOME team")
                with fc2:
                    st.number_input("spread juice", value=-110, step=5,
                                    key=f"spj_{mkey}",
                                    help="Price on the model's side of the "
                                         "spread (usually -110)")
                with fc3:
                    st.number_input("ML price", value=0, step=5,
                                    key=f"mlpick_{mkey}",
                                    help="Price on the model's ML pick")
            st.form_submit_button("Apply odds")

        for g in upcoming:
            mkey = matchup_key(g)

            # ML gate — identical math to the MLB app.
            ml_price = st.session_state.get(f"mlpick_{mkey}", 0)
            if abs(ml_price) >= 100:
                pick_p = (g["home_pct"] if g["prob_pick"] == g["home"]
                          else g["away_pct"]) / 100.0
                be_p = breakeven_prob(ml_price)
                cushion = pick_p - be_p
                ev = pick_p * unit_profit(ml_price) - (1 - pick_p)
                if cushion >= GATE_THRESH_PP / 100.0:
                    call, ccol = "✅ GOOD PICK", "#00c07a"
                elif cushion >= 0:
                    call, ccol = "🟡 THIN — NO BET (edge inside model error)", "#f5c842"
                else:
                    call, ccol = "⛔ STAY AWAY", "#ff5252"
                gate_ml[mkey] = {"pick": g["prob_pick"], "odds": ml_price,
                                 "model": pick_p, "be": be_p,
                                 "cushion": cushion, "ev": ev,
                                 "call": call, "color": ccol}

            # ATS gate — needs a real market spread entered.
            mkt_sp = st.session_state.get(f"sp_{mkey}", 0.0) or 0.0
            sp_juice = st.session_state.get(f"spj_{mkey}", -110)
            has_line = (abs(mkt_sp) >= 0.25 or
                        # PK is a legitimate line but indistinguishable from
                        # "not entered" at exactly 0.0 — require the ESPN
                        # prefill or any nonzero value; PK games are rare
                        # enough to accept entering ±0.5 manually.
                        False)
            if has_line and abs(sp_juice) >= 100 and g["fbs_both"]:
                cp_home = cover_prob_home(g["pred_margin"], mkt_sp)
                if cp_home >= 0.5:
                    ats_side, cp = g["home"], cp_home
                    ats_line = mkt_sp
                else:
                    ats_side, cp = g["away"], 1.0 - cp_home
                    ats_line = -mkt_sp
                be_p = breakeven_prob(sp_juice)
                cushion = cp - be_p
                ev = cp * unit_profit(sp_juice) - (1 - cp)
                if cushion >= GATE_THRESH_PP / 100.0:
                    call, ccol = "✅ GOOD PICK", "#00c07a"
                elif cushion >= 0:
                    call, ccol = "🟡 THIN — NO BET (edge inside model error)", "#f5c842"
                else:
                    call, ccol = "⛔ STAY AWAY", "#ff5252"
                gate_ats[mkey] = {"pick": f"{ats_side} {fmt_spread(ats_line)}",
                                  "side": ats_side, "line": ats_line,
                                  "home_spread": mkt_sp,
                                  "odds": sp_juice, "model": cp, "be": be_p,
                                  "cushion": cushion, "ev": ev,
                                  "call": call, "color": ccol}

        # Auto-sync entered prices into any rows already logged today —
        # the panel is the source of truth for the CURRENT week's prices.
        _synced = 0
        if gate_ml or gate_ats:
            _log = load_pick_log()
            for _r in _log:
                if _r.get("version") != MODEL_VERSION or _r.get("result"):
                    continue
                if _r.get("market") == "ML":
                    _gc = gate_ml.get(_r["matchup"])
                    if _gc and _r.get("odds", 0) != _gc["odds"]:
                        _r["odds"] = _gc["odds"]
                        _r["edge"] = round(_gc["cushion"] * 100, 1)
                        _synced += 1
                elif _r.get("market") == "ATS":
                    _gc = gate_ats.get(_r["matchup"])
                    if _gc and (_r.get("odds", 0) != _gc["odds"]
                                or _r.get("line") != _gc["line"]
                                or _r.get("pick") != _gc["pick"]):
                        _r["odds"] = _gc["odds"]
                        _r["line"] = _gc["line"]
                        _r["pick"] = _gc["pick"]
                        _r["edge"] = round(_gc["cushion"] * 100, 1)
                        _synced += 1
            if _synced:
                save_pick_log(_log)

        # Decision board — both markets, best cushion first.
        board_rows = []
        for k, v in gate_ml.items():
            board_rows.append({"Market": "ML", "Call": v["call"],
                               "Matchup": k, "Pick": v["pick"],
                               "Line": "—", "Price": f"{v['odds']:+d}",
                               "Model %": round(v["model"] * 100, 1),
                               "Needs %": round(v["be"] * 100, 1),
                               "Cushion": f"{v['cushion']*100:+.1f}",
                               "EV/unit": f"{v['ev']*100:+.1f}%",
                               "_c": v["cushion"]})
        for k, v in gate_ats.items():
            board_rows.append({"Market": "ATS", "Call": v["call"],
                               "Matchup": k, "Pick": v["side"],
                               "Line": fmt_spread(v["line"]),
                               "Price": f"{v['odds']:+d}",
                               "Model %": round(v["model"] * 100, 1),
                               "Needs %": round(v["be"] * 100, 1),
                               "Cushion": f"{v['cushion']*100:+.1f}",
                               "EV/unit": f"{v['ev']*100:+.1f}%",
                               "_c": v["cushion"]})
        if board_rows:
            board_rows.sort(key=lambda r: -r["_c"])
            board_df = pd.DataFrame(board_rows).drop(columns=["_c"])
            st.dataframe(board_df, hide_index=True, width="stretch")
            n_good = sum(1 for r in board_rows if r["Call"].startswith("✅"))
            if n_good == 0:
                st.markdown("<div style='font-size:13px;color:#f5c842;"
                            "font-weight:700;'>No plays — sitting out IS "
                            "the play. The market offered nothing.</div>",
                            unsafe_allow_html=True)
            else:
                st.caption(f"{n_good} playable of {len(board_rows)} priced "
                           f"market(s).")
            if _synced:
                st.caption(f"🔄 Synced {_synced} price(s) into logged "
                           f"tracker rows automatically.")

    # ── Render each game ───────────────────────────────────────────────────
    for game in slate_results:
        color = game["conf_color"]
        mkey = matchup_key(game)

        if game["completed"]:
            hs, as_ = game["home_score"], game["away_score"]
            winner = game["home"] if hs > as_ else game["away"]
            status_badge = (
                f"<span style='background:rgba(0,192,122,0.15);color:#00c07a;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>"
                f"Final · {winner} won {max(hs,as_)}-{min(hs,as_)}</span>")
        elif game["state"] == "in":
            status_badge = (
                f"<span style='background:rgba(255,82,82,0.15);color:#ff5252;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>"
                f"🔴 Live · {game['status_detail']}</span>")
        else:
            status_badge = (
                f"<span style='background:rgba(61,139,255,0.15);color:#3d8bff;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>"
                f"{game['kick_label']}</span>")

        neutral_badge = ("<span style='font-size:11px;color:#aaa;"
                         "margin-left:6px;'>🏟 Neutral site — no HFA "
                         "applied</span>" if game["neutral"] else "")

        sh, sa = game["sh"], game["sa"]
        # Data-source tag: mirrors the MLB card's recent/season indicator.
        wp = game["w_prior"]
        if not game["fbs_both"]:
            src_tag = "⚠️ non-FBS opponent"
        elif wp >= 0.99:
            src_tag = f"⚪ 100% {PRIOR_SEASON} data"
        elif wp > 0:
            src_tag = f"🟡 {wp:.0%} {PRIOR_SEASON} / {1-wp:.0%} {SEASON}"
        else:
            src_tag = f"🟢 100% {SEASON} data"
        blend_note = (f"<span style='font-size:10px;color:#666;'>"
                      f"Data: {src_tag}</span>")

        reasons_html = "".join(
            f"<div style='margin:2px 0;font-size:12px;color:#aaa;'>&bull; {r}</div>"
            for r in game["conf_reasons"])

        home_prob_color  = "#00c07a" if game["prob_pick"] == game["home"] else "#aaa"
        home_prob_weight = "800"     if game["prob_pick"] == game["home"] else "400"
        away_prob_color  = "#00c07a" if game["prob_pick"] == game["away"] else "#aaa"
        away_prob_weight = "800"     if game["prob_pick"] == game["away"] else "400"

        rec_h = f"{sh.get('wins',0)}-{sh.get('losses',0)}" if game["fbs_both"] else game["home_record"]
        rec_a = f"{sa.get('wins',0)}-{sa.get('losses',0)}" if game["fbs_both"] else game["away_record"]
        rec_h = game["home_record"] or rec_h
        rec_a = game["away_record"] or rec_a

        card_html = (
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;'
            f'border-left:4px solid {color};border-radius:12px;padding:18px 22px;margin-bottom:16px;">'

            # Header
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'flex-wrap:wrap;gap:8px;margin-bottom:14px;">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
            f'<span style="font-size:17px;font-weight:700;">'
            f'{game["away"]} <span style="color:#555;font-size:13px;font-weight:400;">({rec_a})</span>'
            f' <span style="color:#555;margin:0 6px;">@</span> '
            f'{game["home"]} <span style="color:#555;font-size:13px;font-weight:400;">({rec_h})</span>'
            f'</span> {status_badge} {neutral_badge}</div>'
            f'<div style="margin-top:4px;">{blend_note}</div>'
            f'<span style="font-size:13px;font-weight:700;color:{color};">'
            f'{game["conf_emoji"]} {game["conf_level"]} confidence</span></div>'

            # Stats grid
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:14px;">'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Win probability</div>'
            f'<div style="font-size:14px;">'
            f'<span style="color:{home_prob_color};font-weight:{home_prob_weight};">'
            f'{game["home"]} {game["home_pct"]}%</span><br>'
            f'<span style="color:{away_prob_color};font-weight:{away_prob_weight};">'
            f'{game["away"]} {game["away_pct"]}%</span></div></div>'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Projected score</div>'
            f'<div style="font-size:14px;font-weight:600;">'
            f'{game["home"]} <span style="color:#f5c842;">{game["proj_home"]}</span><br>'
            f'{game["away"]} <span style="color:#f5c842;">{game["proj_away"]}</span><br>'
            f'<span style="font-size:11px;color:#888;font-weight:400;">O/U: '
            f'<span style="color:#f5c842;">{game["proj_total"]}</span></span></div></div>'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Model line</div>'
            f'<div style="font-size:14px;font-weight:700;color:{color};">'
            f'{game["home"]} {fmt_spread(game["model_line_home"])}'
            f'<br><span style="font-size:11px;font-weight:400;color:#888;">'
            f'Proj margin: {abs(game["pred_margin"]):.1f} pts</span></div></div>'

            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Key stats · opp-adjusted (H / A)</div>'
            f'<div style="font-size:11px;color:#aaa;">'
            f'PPG: <span style="color:#ccc;">{sh["pf_pg"]:.1f} / {sa["pf_pg"]:.1f}</span><br>'
            f'PPG allowed: <span style="color:#ccc;">{sh["pa_pg"]:.1f} / {sa["pa_pg"]:.1f}</span><br>'
            f'Margin/G: <span style="color:#ccc;">{sh["mpg"]:+.1f} / {sa["mpg"]:+.1f}</span><br>'
            f'W%: <span style="color:#ccc;">{sh["wpct"]:.0f}% / {sa["wpct"]:.0f}%</span><br>'
            f'Sched adj: <span style="color:#ccc;">{sh.get("sos", 0):+.1f} / {sa.get("sos", 0):+.1f}</span>'
            f'</div></div></div>'

            # Reasons
            f'<div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:10px;">'
            f'{reasons_html}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Gate verdicts under the card (read-only; entry is in the panel)
        if not game["completed"]:
            lines = []
            gc = gate_ml.get(mkey)
            if gc:
                lines.append(
                    f"<span style='color:#888;'>ML&nbsp;</span>"
                    f"<span style='color:{gc['color']};font-weight:700;'>{gc['call']}</span>"
                    f"<span style='color:#888;'> at {gc['odds']:+d} — model "
                    f"{gc['model']*100:.1f}% vs needed {gc['be']*100:.1f}% "
                    f"(cushion {gc['cushion']*100:+.1f}pp, "
                    f"EV {gc['ev']*100:+.1f}%/unit)</span>")
            ga = gate_ats.get(mkey)
            if ga:
                lines.append(
                    f"<span style='color:#888;'>ATS&nbsp;</span>"
                    f"<span style='color:{ga['color']};font-weight:700;'>{ga['call']}</span>"
                    f"<span style='color:#888;'> {ga['pick']} at {ga['odds']:+d} — "
                    f"cover {ga['model']*100:.1f}% vs needed {ga['be']*100:.1f}% "
                    f"(cushion {ga['cushion']*100:+.1f}pp, "
                    f"EV {ga['ev']*100:+.1f}%/unit)</span>")
            if lines:
                st.markdown("<div style='font-size:12px;margin:-6px 0 14px 2px;'>"
                            + "<br>".join(lines) + "</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='font-size:11px;color:#555;"
                    "margin:-6px 0 14px 2px;'>No prices entered — add them "
                    "in the price gate panel above for go/no-go calls</div>",
                    unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PICK TRACKER — forward measurement, two markets per game.
    # Same instrument as the MLB tracker: logs the live model's picks
    # exactly as displayed (no look-ahead possible), tagged with
    # MODEL_VERSION, every game on the slate — the un-bet games are the
    # control group. ML and ATS are separate rows and are summarized
    # separately: the model can be good at one and bad at the other, and
    # only per-market samples can tell.
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📌 Pick Tracker")
    _cfg = _gh_cfg()
    if _cfg:
        st.caption(f"🗄 Log storage: GitHub — {_cfg['repo']} @ {_cfg['branch']} "
                   f"(survives Streamlit Cloud reboots)")
    else:
        st.caption("🗄 Log storage: local file — fine on your machine, but "
                   "WIPED on Streamlit Cloud reboots. Add [github] secrets "
                   "to persist (see comment above _gh_cfg in the code).")

    log = load_pick_log()
    existing_keys = {(r["date"], r["matchup"], r.get("market", "ML"))
                     for r in log}

    c_log1, c_log3, c_log2 = st.columns([1, 1, 1.4])
    with c_log1:
        if st.button("Log this week's slate to tracker"):
            added = 0
            skipped_final = 0
            for g in slate_results:
                # Already-final games are EXCLUDED on purpose: a pick logged
                # after the result exists is look-ahead (the model's stats
                # already contain that game, and the outcome is known).
                # The forward record only admits games that could still be
                # bet at log time — but the skip is announced, not silent.
                if g["completed"]:
                    skipped_final += 1
                    continue
                matchup = matchup_key(g)
                row_date = g["kick_date"] or today_et()
                # ML row for every game.
                if (row_date, matchup, "ML") not in existing_keys:
                    gc = gate_ml.get(matchup)
                    log.append({
                        "date": row_date, "matchup": matchup, "market": "ML",
                        "pick": g["prob_pick"],
                        "line": None,
                        "prob": round(max(g["home_pct"], g["away_pct"]), 1),
                        "tier": g["conf_level"], "version": MODEL_VERSION,
                        "odds": gc["odds"] if gc else 0,
                        "edge": round(gc["cushion"] * 100, 1) if gc else None,
                        "result": "",
                    })
                    added += 1
                # ATS row only once a market line exists — a spread pick
                # without its line is ungradeable.
                ga = gate_ats.get(matchup)
                if ga and (row_date, matchup, "ATS") not in existing_keys:
                    log.append({
                        "date": row_date, "matchup": matchup, "market": "ATS",
                        "pick": ga["pick"],
                        "line": ga["line"],
                        "prob": round(ga["model"] * 100, 1),
                        "tier": g["conf_level"], "version": MODEL_VERSION,
                        "odds": ga["odds"],
                        "edge": round(ga["cushion"] * 100, 1),
                        "result": "",
                    })
                    added += 1
            save_pick_log(log)
            msg = (f"Logged {added} new pick rows."
                   if added else "This week's slate is already logged — "
                                 "prices entered above sync into it "
                                 "automatically.")
            if skipped_final:
                msg += (f" Skipped {skipped_final} already-final game(s): "
                        f"picks can't be logged after the result exists "
                        f"(look-ahead) — only games still biddable at log "
                        f"time enter the forward record.")
            msg += (" ATS rows appear only for games with a spread entered "
                    "— enter spreads above and click Log again to add them.")
            st.success(msg)
            st.rerun()
    with c_log3:
        # Auto-grading pulls finals from the same ESPN API as the slate.
        # ML: winner vs pick. ATS: final margin vs the LOGGED line —
        # margin exactly on the number → Push. Postponed/canceled → Push.
        if st.button("⚡ Auto-grade finished games"):
            graded = 0
            today_iso = today_et("%Y-%m-%d")
            for r in log:
                if r.get("result") or r["date"] > today_iso:
                    continue
                if " @ " not in r["matchup"]:
                    continue
                try:
                    day_games = fetch_results_for_date(r["date"])
                except Exception:
                    continue
                away, home = r["matchup"].split(" @ ", 1)
                gm = next((x for x in day_games
                           if x["away"] == away and x["home"] == home), None)
                # Late kickoffs that cross midnight ET can be filed under
                # the neighboring calendar day on ESPN's scoreboard — check
                # the adjacent days before giving up on the row.
                if gm is None:
                    base = datetime.datetime.strptime(r["date"], "%Y-%m-%d")
                    for delta in (-1, 1):
                        alt = (base + datetime.timedelta(days=delta)
                               ).strftime("%Y-%m-%d")
                        try:
                            alt_games = fetch_results_for_date(alt)
                        except Exception:
                            continue
                        gm = next((x for x in alt_games
                                   if x["away"] == away and x["home"] == home),
                                  None)
                        if gm is not None:
                            break
                if gm is None:
                    continue
                if gm["completed"]:
                    hs, as_ = gm["home_score"], gm["away_score"]
                    if r.get("market", "ML") == "ML":
                        winner = home if hs > as_ else away
                        r["result"] = "W" if winner == r["pick"] else "L"
                        graded += 1
                    else:   # ATS
                        line = r.get("line")
                        if line is None:
                            continue
                        # r["pick"] is "Team ±X.X" — side name is the part
                        # before the trailing spread token.
                        side = r["pick"].rsplit(" ", 1)[0]
                        side_margin = (hs - as_) if side == home else (as_ - hs)
                        adj = side_margin + float(line)
                        r["result"] = ("Push" if abs(adj) < 1e-9
                                       else "W" if adj > 0 else "L")
                        # CLV: ESPN's last carried line at grading time,
                        # from the pick side's perspective. Positive CLV =
                        # the logged number was BETTER than where the
                        # market ended up — the fastest-converging evidence
                        # of real edge (long before W/L records mean
                        # anything). Best-effort: ESPN doesn't carry a line
                        # on every final.
                        close_home = (gm.get("odds") or {}).get("home_spread")
                        if close_home is not None:
                            close_side = (float(close_home) if side == home
                                          else -float(close_home))
                            r["close"] = close_side
                            r["clv"] = round(float(line) - close_side, 1)
                        graded += 1
                elif "postponed" in gm["status_detail"].lower() or \
                        "cancel" in gm["status_detail"].lower():
                    r["result"] = "Push"
                    graded += 1
            if graded:
                save_pick_log(log)
                st.success(f"Auto-graded {graded} picks.")
                st.rerun()
            else:
                st.info("Nothing new to grade yet.")
    with c_log2:
        st.caption("Weekly loop: enter prices in the panel, log the slate "
                   "(2 rows per priced game — ML and ATS), auto-grade "
                   "after the games. Every game gets graded — bet or not. "
                   "Verdicts need ~100 graded picks per market per tier.")

    if log:
        df = pd.DataFrame(log)
        if "market" not in df.columns:
            df["market"] = "ML"
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False,
                                kind="stable").reset_index(drop=True)
        edited = st.data_editor(
            df,
            column_config={
                "odds":   st.column_config.NumberColumn(
                    "odds", help="American price you actually got", step=5),
                "result": st.column_config.SelectboxColumn(
                    "result", options=["", "W", "L", "Push"]),
                "date":    st.column_config.TextColumn(disabled=True),
                "matchup": st.column_config.TextColumn(disabled=True),
                "market":  st.column_config.TextColumn(disabled=True),
                "pick":    st.column_config.TextColumn(disabled=True),
                "line":    st.column_config.NumberColumn(
                    "line", help="Logged market spread for the pick side "
                    "(ATS rows only) — grading uses THIS number", step=0.5),
                "prob":    st.column_config.NumberColumn(disabled=True),
                "tier":    st.column_config.TextColumn(disabled=True),
                "version": st.column_config.TextColumn(disabled=True),
                "edge":    st.column_config.NumberColumn(
                    "edge", help="Cushion: model prob − break-even prob of "
                    "the logged price (pp)", disabled=True),
                "close":   st.column_config.NumberColumn(
                    "close", help="ESPN's last carried line for the pick "
                    "side at grading time", disabled=True),
                "clv":     st.column_config.NumberColumn(
                    "clv", help="Closing line value in points: logged line "
                    "− close, pick-side view. Positive = beat the close.",
                    disabled=True),
            },
            num_rows="dynamic",
            hide_index=True, width="stretch", height=300)
        if st.button("Save grades"):
            save_pick_log(edited.to_dict("records"))
            st.success("Saved.")
            st.rerun()

        # Summary — current model version only, graded picks only,
        # SPLIT BY MARKET first: ML skill and ATS skill are different
        # claims and get different verdicts.
        cur = edited[(edited["version"] == MODEL_VERSION) &
                     (edited["result"].isin(["W", "L"]))]
        if len(cur):
            st.markdown(f"**{MODEL_VERSION}** — graded picks: {len(cur)}")
            sum_rows = []
            for market in ["ML", "ATS"]:
                m = cur[cur["market"] == market]
                for tier in ["High", "Moderate", "Low", "Conflicted"]:
                    t = m[m["tier"] == tier]
                    if not len(t):
                        continue
                    w = int((t["result"] == "W").sum())
                    l = int((t["result"] == "L").sum())
                    priced = t[t["odds"] != 0]
                    units = sum(unit_profit(r["odds"]) if r["result"] == "W"
                                else -1.0 for _, r in priced.iterrows())
                    sum_rows.append({
                        "market": market, "tier": tier,
                        "record": f"{w}–{l}",
                        "hit %": round(w / (w + l) * 100, 1),
                        "units (priced picks)": round(units, 2),
                        "toward 100": f"{w + l}/100",
                    })
            if sum_rows:
                st.dataframe(pd.DataFrame(sum_rows), hide_index=True,
                             width="stretch")

            # Cushion buckets per market — the data that decides where the
            # GOOD threshold belongs, and for ATS whether SIGMA_COVER is
            # honest: "model says" far above "actual" in a bucket means
            # the cover probabilities are overconfident there.
            if "edge" in cur.columns:
                brows = []
                for market in ["ML", "ATS"]:
                    priced = cur[(cur["market"] == market) &
                                 (cur["odds"] != 0) & cur["edge"].notna()]
                    buckets = [("cushion ≥ +3pp", priced["edge"] >= 3.0),
                               ("cushion 0 to +3pp",
                                (priced["edge"] >= 0) & (priced["edge"] < 3.0)),
                               ("cushion < 0 (stay-aways, if bet anyway)",
                                priced["edge"] < 0)]
                    for label, mask in buckets:
                        b = priced[mask]
                        if not len(b):
                            continue
                        w = int((b["result"] == "W").sum())
                        l = int((b["result"] == "L").sum())
                        u = sum(unit_profit(r["odds"]) if r["result"] == "W"
                                else -1.0 for _, r in b.iterrows())
                        avg_model = b["prob"].mean()
                        avg_need = (b["prob"] - b["edge"]).mean()
                        brows.append({"market": market,
                                      "cushion bucket": label,
                                      "record": f"{w}–{l}",
                                      "model says": f"{avg_model:.1f}%",
                                      "price needs": f"{avg_need:.1f}%",
                                      "actual": f"{w/(w+l)*100:.1f}%",
                                      "units": round(u, 2),
                                      "toward 100": f"{w+l}/100"})
                if brows:
                    st.markdown("**Cushion buckets** — per market. Compare "
                                "*model says* vs *actual*: a gap is "
                                "overconfidence in that zone (for ATS that "
                                "points at SIGMA_COVER).")
                    st.dataframe(pd.DataFrame(brows), hide_index=True,
                                 width="stretch")
            ats_clv = cur[(cur["market"] == "ATS")] if "market" in cur.columns else cur.iloc[0:0]
            if "clv" in ats_clv.columns:
                ats_clv = ats_clv[ats_clv["clv"].notna()]
                if len(ats_clv):
                    overall = ats_clv["clv"].mean()
                    good = ats_clv[ats_clv["edge"].fillna(-99) >= 3.0]
                    msg = (f"**Closing line value (ATS):** {overall:+.2f} "
                           f"pts/pick over {len(ats_clv)} picks")
                    if len(good):
                        msg += (f" · GOOD-bucket picks: "
                                f"{good['clv'].mean():+.2f} over {len(good)}")
                    msg += (". Sustained positive CLV is edge evidence that "
                            "converges far faster than W/L; sustained "
                            "negative CLV on GOOD picks is an early warning "
                            "the model is chasing stale numbers.")
                    st.markdown(msg)
            st.caption("Units use only picks with a price entered, flat 1u. "
                       "ATS hit rate must clear ~52.4% at -110 just to break "
                       "even — treat anything under 55% over a small sample "
                       "as noise.")
    else:
        st.caption("No picks logged yet. Log this week's slate to start "
                   "the record.")

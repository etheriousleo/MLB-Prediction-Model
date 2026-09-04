"""
NFL Weekly Slate — Win Probability & Spreads
--------------------------------------------
NFL sibling of the CFB Weekly Slate and MLB Daily Slate apps. Same
philosophy: frozen model, hardcoded price gate, full-slate pick tracker
with auto-grading, GitHub-backed persistence.

Install dependencies:
    pip install streamlit pandas numpy requests

Run:
    streamlit run nfl_app.py

Data source: ESPN's public NFL API — no API key needed. Team stats (PPG
for/against, margin, win%) are computed from actual game scores pulled
week by week, then opponent-adjusted (SRS-style), so the whole app is
keyless like its siblings.

What's structurally different from the CFB app, and why:
  * No FBS/FCS membership layer — 32 teams, every game is readable.
  * Ties exist. Stats, records, and moneyline grading all handle them
    (an ML pick on a tied game is a Push, not a loss).
  * International games (nine in 2026) strip home-field advantage from
    the designated "home" team — ESPN's neutral flag plus a venue check.
  * NFL-calibrated constants: HFA 1.5, margin sigma ~14, a tighter
    z-to-points multiplier, prior-season fade over 8 games.
  * Playoff rounds are selectable from day one (a slate selector, not a
    model change — freeze-compatible).
  * Closing line value is captured for BOTH markets: ATS in points,
    ML in probability points, from ESPN's final carried odds.
"""

import base64
import datetime
import json
import math
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from zoneinfo import ZoneInfo

st.set_page_config(page_title="NFL Weekly Slate", page_icon="🏈", layout="wide")

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


# NFL season year = the calendar year of the fall the season starts in.
# (January/February playoffs still belong to the previous year's season;
# ESPN's API uses the same convention, so this stays correct through the
# Super Bowl.)
_now = now_et()
SEASON = _now.year if _now.month >= 6 else _now.year - 1
PRIOR_SEASON = SEASON - 1

# ── MODEL FREEZE ───────────────────────────────────────────────────────────────
# Same discipline as the MLB and CFB apps: this version is FROZEN for
# forward measurement. Every logged pick is tagged with this string. No
# parameter/formula changes until the tracker holds ≥100 graded picks
# PER MARKET (ML and ATS separately) for this version.
MODEL_VERSION = "nfl-v1.0-frozen-2026-09-04"

# ── GATE THRESHOLD — deliberately NOT adjustable in the UI ─────────────────────
# Carried over verbatim from the MLB/CFB apps at Juan's request: no in-app
# knob, so in-the-moment eagerness can't loosen the discipline. Changing
# this requires editing code — that friction is the feature. Same 3.0pp
# cushion applies to BOTH markets: for ML the model's win prob must clear
# the price's break-even by 3pp; for ATS the model's cover prob must clear
# the juice's break-even (52.4% at -110) by 3pp — which at SIGMA_COVER
# works out to the model's line disagreeing with the market by ~2 points.
# The NFL is the sharpest market there is: expect FEWER good verdicts
# than the CFB app produces. That is the gate working, not failing.
GATE_THRESH_PP = 3.0

# ── Reference-line ATS control rows — OFF, matching the CFB decision ──────────
# False = ESPN lines never enter the tracker; every ATS row exists only
# because the user entered that game's spread by hand (or is pending the
# user's line). Set True to have unpriced games logged against ESPN's
# carried line as odds-0 control rows.
REFERENCE_ATS_ROWS = False

# ── Margin model constants ─────────────────────────────────────────────────────
# HFA: NFL home-field advantage has shrunk hard over the last decade —
# home teams win ~52–55% straight up in recent seasons, which at a 13.5-
# point margin sigma implies ~1–1.7 points. 1.5 is the anchored public
# base rate, not a tuned value. Neutral/international games get 0.
HFA_POINTS = 1.5

# K_MARGIN: converts the composite z-score gap into points. With the NFL
# anchors below, the composite works out to ≈0.12 × (opponent-adjusted
# margin per game), so K=7.5 recovers ~90% of the raw SRS-style rating
# gap — a mild built-in regression toward the mean, which is appropriate
# because observed margins overstate true strength. Best-vs-worst NFL
# mismatches project to ~20 points before HFA, in line with the widest
# real NFL spreads. Reasoned default, not a fitted value.
K_MARGIN = 7.5

# SIGMA_WIN: std dev of actual margin around the true expected margin,
# used to convert projected margin → win probability via the normal CDF.
# NFL finals scatter around closing spreads with σ ≈ 13.5 — the most
# thoroughly measured number in sports betting. Our projection is noisier
# than a closing line, so 14.0 adds a small model-error allowance.
SIGMA_WIN = 14.0

# SIGMA_COVER: same distribution drives cover probability. Kept as its own
# named constant because cover math is the part most worth revisiting once
# the ATS tracker has data. KNOWN LIMITATION, stated up front: NFL margins
# are NOT normal — they pile up on key numbers (3, 7, 6, 10, 4). The normal
# CDF understates push/cover odds right around those numbers. If the
# cushion buckets show ATS overconfidence, this is where to look.
SIGMA_COVER = 14.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── ESPN endpoints ─────────────────────────────────────────────────────────────
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
REG_WEEKS = 18            # 2021+ regular season: 18 weeks, 17 games per team
POST_ROUNDS = {1: "Wild Card", 2: "Divisional", 3: "Conference", 5: "Super Bowl"}
# (ESPN's postseason "week 4" is the Pro Bowl slot — never a real game.)

# International venues: the designated home team gets no HFA. ESPN flags
# some of these as neutralSite; the venue check catches the rest. Cities
# rather than stadiums so a new venue in a known city still matches.
INTL_CITIES = {"london", "munich", "frankfurt", "berlin", "madrid", "paris",
               "melbourne", "sydney", "mexico city", "sao paulo",
               "são paulo", "rio de janeiro", "dublin", "toronto"}


def _espn_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _parse_event(ev: dict) -> dict | None:
    """Flatten one ESPN scoreboard event into the fields we use."""
    try:
        comp = ev["competitions"][0]
        home = away = None
        for c in comp.get("competitors", []):
            side = {"id": str(c["team"]["id"]),
                    "name": c["team"].get("displayName", ""),
                    "abbr": c["team"].get("abbreviation", ""),
                    "score": int(float(c.get("score") or 0)),
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
        # Market odds, when ESPN carries them: spread is HOME-relative,
        # plus each side's moneyline. Never prefilled into the gate —
        # their only role is the CLV reference captured at grading time.
        odds = {}
        for o in comp.get("odds", []) or []:
            if "spread" in o or "overUnder" in o or "homeTeamOdds" in o:
                hto = o.get("homeTeamOdds") or {}
                ato = o.get("awayTeamOdds") or {}
                odds = {"home_spread": o.get("spread"),
                        "over_under": o.get("overUnder"),
                        "details": o.get("details", ""),
                        "home_ml": hto.get("moneyLine"),
                        "away_ml": ato.get("moneyLine")}
                break
        venue = comp.get("venue", {}) or {}
        addr = venue.get("address", {}) or {}
        country = str(addr.get("country") or "").strip().upper()
        city = str(addr.get("city") or "").strip().lower()
        intl = ((bool(country) and country not in ("USA", "US", "UNITED STATES"))
                or city in INTL_CITIES)
        neutral = bool(comp.get("neutralSite", False)) or intl
        date_utc = ev.get("date", "")
        try:
            dt = datetime.datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
            dt_et = dt.astimezone(APP_TZ)
            kick_label = dt_et.strftime("%a %I:%M %p ET").replace(" 0", " ")
            kick_date = dt_et.strftime("%Y-%m-%d")
        except Exception:
            kick_label, kick_date = "", ""
        return {
            "event_id": str(ev.get("id", "")),
            "home_id": home["id"], "home": home["name"], "home_abbr": home["abbr"],
            "home_score": home["score"], "home_record": home["record"],
            "away_id": away["id"], "away": away["name"], "away_abbr": away["abbr"],
            "away_score": away["score"], "away_record": away["record"],
            "neutral": neutral, "intl": intl,
            "completed": bool(status.get("completed", False)),
            "state": status.get("state", ""),          # pre / in / post
            "status_detail": status.get("shortDetail", ""),
            "kick_label": kick_label, "kick_date": kick_date,
            "venue": venue.get("fullName", ""),
            "odds": odds,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_week_slate(season: int, week: int, seasontype: int = 2) -> list:
    """All scoreboard events for one week. seasontype 2 = regular season,
    3 = postseason (week = round number, see POST_ROUNDS)."""
    j = _espn_get(f"{ESPN}/scoreboard",
                  {"limit": 100, "dates": season,
                   "seasontype": seasontype, "week": week})
    games = [_parse_event(ev) for ev in j.get("events", [])]
    return [g for g in games if g]


@st.cache_data(ttl=1800)
def fetch_current_week(season: int) -> tuple[int, int]:
    """ESPN's notion of (seasontype, week). Preseason or anything odd
    falls back to regular-season week 1."""
    try:
        j = _espn_get(f"{ESPN}/scoreboard", {"limit": 1})
        wk = j.get("week", {}).get("number")
        yr = j.get("season", {}).get("year")
        stype = j.get("season", {}).get("type")
        if wk and yr == season:
            if stype == 2:
                return 2, max(1, min(REG_WEEKS, int(wk)))
            if stype == 3 and int(wk) in POST_ROUNDS:
                return 3, int(wk)
    except Exception:
        pass
    return 2, 1


@st.cache_data(ttl=600)
def fetch_results_for_date(date_iso: str) -> list:
    """Final scores for all games on a date (for auto-grading)."""
    ymd = date_iso.replace("-", "")
    j = _espn_get(f"{ESPN}/scoreboard", {"limit": 100, "dates": ymd})
    games = [_parse_event(ev) for ev in j.get("events", [])]
    return [g for g in games if g]


# ── Team season stats, computed from game scores ───────────────────────────────
# Same construction as the CFB app: pull every completed game's score and
# derive PPG for/against, margin/game, and win% per team, then opponent-
# adjust. The NFL version has no non-FBS filter (every opponent is one of
# the 32) but must handle ties — an NFL game can end level, and a tie is
# half a win in the record and a zero-margin game in the scoring lines.

@st.cache_data(ttl=1800)
def fetch_season_team_stats(season: int, upto_week: int = REG_WEEKS,
                            post_upto: int = 0) -> dict:
    """
    {team_id: {pf_pg, pa_pg, mpg, wpct, wins, losses, ties, games, sos}}
    from completed games: regular-season weeks 1..upto_week, plus
    postseason rounds 1..post_upto (skipping the Pro Bowl slot) when
    post_upto > 0. Cached 30 min; the prior season's call is effectively
    static.
    """
    acc = {}      # id -> [pf, pa, w, l, t, g]
    gamelog = {}  # id -> [(opp_id, pf, pa)] — feeds the opponent adjustment

    def bump(tid, opp, pf, pa):
        a = acc.setdefault(tid, [0, 0, 0, 0, 0, 0])
        a[0] += pf; a[1] += pa
        if pf > pa:
            a[2] += 1
        elif pf < pa:
            a[3] += 1
        else:
            a[4] += 1
        a[5] += 1
        gamelog.setdefault(tid, []).append((opp, pf, pa))

    schedule = [(2, wk) for wk in range(1, min(REG_WEEKS, upto_week) + 1)]
    schedule += [(3, rd) for rd in range(1, post_upto + 1) if rd in POST_ROUNDS]
    for stype, wk in schedule:
        try:
            slate = fetch_week_slate(season, wk, stype)
        except Exception:
            continue
        for g in slate:
            if not g["completed"]:
                continue
            hs, as_ = g["home_score"], g["away_score"]
            bump(g["home_id"], g["away_id"], hs, as_)
            bump(g["away_id"], g["home_id"], as_, hs)

    # Keyed by ESPN team ID — never display name (the CFB app learned that
    # names can differ across ESPN endpoints; IDs don't).
    out = {}
    for tid, (pf, pa, w, l, t, gp) in acc.items():
        if gp == 0:
            continue
        out[tid] = {
            "pf_pg": pf / gp, "pa_pg": pa / gp,
            "mpg": (pf - pa) / gp,
            "wpct": (w + 0.5 * t) / gp * 100,
            "wins": w, "losses": l, "ties": t, "games": gp,
        }
    return opponent_adjust(out, gamelog)


def opponent_adjust(stats: dict, gamelog: dict, n_iter: int = 30,
                    damp: float = 0.7) -> dict:
    """
    SRS-style iterative opponent adjustment, identical to the CFB app's.
    Each pass re-values every team's offense and defense against its
    opponents' CURRENT adjusted ratings:

        adj_off(i) = mean over i's games of (pts scored − adj_def(opp))
                     + league mean
        adj_def(i) = mean over i's games of (pts allowed − adj_off(opp))
                     + league mean          (lower = better defense)

    Damped and re-centered each pass so early-season sparse schedules
    can't oscillate or drift. The NFL matters MORE for this than CFB:
    a 17-game schedule against a tightly bunched league means a couple
    of easy opponents can swing raw PPG by several points. wpct stays
    RAW (a record is a record); mpg is recomputed from the adjusted
    scoring lines. Clamp band = 3σ of the NFL team-scoring spread.
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
            o = sum(pf - adj_def.get(opp, mean_def)
                    for opp, pf, pa in games) / len(games) + mean_def
            d = sum(pa - adj_off.get(opp, mean_off)
                    for opp, pf, pa in games) / len(games) + mean_off
            new_off[t] = damp * o + (1 - damp) * adj_off[t]
            new_def[t] = damp * d + (1 - damp) * adj_def[t]
        off_shift = mean_off - sum(new_off.values()) / len(new_off)
        def_shift = mean_def - sum(new_def.values()) / len(new_def)
        adj_off = {t: v + off_shift for t, v in new_off.items()}
        adj_def = {t: v + def_shift for t, v in new_def.items()}
    band = 3 * LEAGUE_STD["pf_pg"]
    lo_o, hi_o = mean_off - band, mean_off + band
    lo_d, hi_d = mean_def - band, mean_def + band
    for t, s in stats.items():
        s["raw_pf_pg"], s["raw_pa_pg"] = s["pf_pg"], s["pa_pg"]
        s["pf_pg"] = min(hi_o, max(lo_o, adj_off[t]))
        s["pa_pg"] = min(hi_d, max(lo_d, adj_def[t]))
        s["mpg"] = s["pf_pg"] - s["pa_pg"]
        s["sos"] = round((s["pf_pg"] - s["raw_pf_pg"])
                         + (s["raw_pa_pg"] - s["pa_pg"]), 1)
    return stats


# League anchors for z-scoring. NFL scoring runs ~22–23 PPG with a ~4.5
# point std across teams; margin/game std ~6.5 (the best teams live near
# +12, the worst near −12); win% std ~19pp (3-14 to 15-2 in a 17-game
# season). Fixed anchors (like the MLB/CFB LEAGUE_AVG tables) so a
# drifting league mean can't silently rescale every rating mid-season.
LEAGUE_AVG = {"pf_pg": 22.5, "pa_pg": 22.5, "mpg": 0.0, "wpct": 50.0}
LEAGUE_STD = {"pf_pg": 4.5,  "pa_pg": 4.5,  "mpg": 6.5, "wpct": 19.0}


def zscore(val: float, key: str, inv: bool = False) -> float:
    z = (val - LEAGUE_AVG[key]) / LEAGUE_STD[key]
    z = max(-2.5, min(2.5, z))
    return -z if inv else z


def blend_prior(cur: dict | None, prior: dict | None, fade_games: int) -> tuple[dict | None, float]:
    """
    Same backward-pointing blend as the CFB app: at 0 current-season
    games the model runs 100% on LAST season's numbers, fading linearly
    to 100% current by `fade_games`. This is what makes Week 1 usable.
    NFL year-over-year continuity is stronger than CFB's (no transfer
    portal) but still modest — QB changes, coaching turnover, and the
    cap churn a third of every roster — so the default fade is 8 games,
    roughly the point where public preseason projections and in-season
    results carry equal weight. Returns (blended_stats, prior_weight).
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
    blended["wins"] = cur.get("wins", 0)
    blended["losses"] = cur.get("losses", 0)
    blended["ties"] = cur.get("ties", 0)
    return blended, w_prior


def build_components(s: dict) -> tuple[float, float, float]:
    """(offense, defense, record) component z-scores for one team.
    Record composite keeps the siblings' W% 25 / margin 75 split: NFL
    close-game records are close to coin flips, so margin per game — the
    far less luck-contaminated signal — carries the weight."""
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
    Probability depends only on the GAP via the margin distribution —
    never on an absolute level that could drift."""
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


PENDING_ATS_PICK = "— enter HOME spread in line —"


def resolve_ats_rows(rows: list) -> int:
    """
    Resolve pending ATS rows once the user has typed a line into the
    tracker. A pending row carries the model's projected HOME margin,
    PRE-REGISTERED at log time — so when a HOME spread appears in its
    line column, the model's side follows mechanically (no judgment, no
    look-ahead possible): home covers iff margin beats the number.
    The line is then restated for the pick side (grading convention).
    Returns how many rows changed. Graded rows are never touched.
    """
    changed = 0
    for r in rows:
        if (r.get("market") == "ATS" and not r.get("result")
                and r.get("pick") == PENDING_ATS_PICK
                and r.get("line") is not None
                and r.get("margin") is not None):
            try:
                home_sp = float(r["line"])
                margin = float(r["margin"])
            except (TypeError, ValueError):
                continue
            away, home = r["matchup"].split(" @ ", 1)
            cp_home = cover_prob_home(margin, home_sp)
            if cp_home >= 0.5:
                side, cp, ln = home, cp_home, home_sp
            else:
                side, cp, ln = away, 1 - cp_home, -home_sp
            r["pick"] = f"{side} {fmt_spread(ln)}"
            r["line"] = ln
            r["prob"] = round(cp * 100, 1)
            odds = r.get("odds", 0) or 0
            if abs(odds) >= 100:
                r["edge"] = round((cp - breakeven_prob(odds)) * 100, 1)
            changed += 1
    return changed


def matchup_key(g: dict) -> str:
    """Unique, human-readable identity for a game. No doubleheaders in
    the NFL, so away @ home is already unique within a date."""
    return f"{g['away']} @ {g['home']}"


def fmt_spread(x: float) -> str:
    """Display a home-relative spread the way books quote it."""
    if abs(x) < 0.25:
        return "PK"
    return f"{x:+g}"


def fmt_record(s: dict) -> str:
    w, l, t = s.get("wins", 0), s.get("losses", 0), s.get("ties", 0)
    return f"{w}-{l}-{t}" if t else f"{w}-{l}"


def calc_confidence(g: dict, sh: dict, sa: dict, w_prior: float,
                    pct_h: float, pct_a: float, pred_margin: float,
                    readable: bool) -> tuple[str, str, str, list]:
    """Mirrors the sibling tiers: High / Moderate / Low, with
    plain-language reasons. Thresholds map back through SIGMA_WIN and
    land on NFL key numbers: strong ≥35pp is a ~6.5+ point projected
    edge (roughly a touchdown favorite), moderate ≥15pp is ~2.5+ (a
    field goal). Prior-season reliance and low game counts cap the tier."""
    prob_winner = g["home"] if pct_h >= pct_a else g["away"]
    prob_gap = abs(pct_h - pct_a)
    prob_strength = ("strong" if prob_gap >= 35
                     else "moderate" if prob_gap >= 15 else "narrow")

    # Scoring leader vs record leader — usually schedule, sometimes luck
    # in one-score games. Either way it's a reason to trust the read less.
    score_leader = g["home"] if (sh["pf_pg"] - sh["pa_pg"]) > (sa["pf_pg"] - sa["pa_pg"]) else g["away"]
    rec_leader = g["home"] if sh["wpct"] > sa["wpct"] else g["away"]
    split = score_leader != rec_leader

    low_data = min(sh.get("games", 0), sa.get("games", 0)) < 3
    prior_heavy = w_prior >= 0.5

    if not readable:
        level, emoji, color = "Low", "🟠", "#f5a623"
    elif prob_strength == "strong" and not split and not prior_heavy and not low_data:
        level, emoji, color = "High", "🟢", "#00c07a"
    elif prob_strength in ("strong", "moderate"):
        level, emoji, color = "Moderate", "🟡", "#f5c842"
    else:
        level, emoji, color = "Low", "🟠", "#f5a623"

    reasons = []
    if not readable:
        reasons.append("A team here has **no games in either season's data** "
                       "— a placeholder rating is in use, so the model output "
                       "is not a read. Treat as no-play.")
    if prior_heavy:
        reasons.append(f"Model is running **{w_prior:.0%} on last season's "
                       f"data** — offseason turnover (QB, coaching, a third of "
                       f"the roster) makes early-season projections carry "
                       f"extra uncertainty by design.")
    elif low_data:
        reasons.append("Fewer than 3 current-season games for at least one "
                       "team — small sample, and the schedule adjustment is "
                       "barely constrained yet.")
    if split:
        reasons.append(f"**{score_leader}** has the better scoring margin but "
                       f"**{rec_leader}** has the better record — schedule or "
                       f"one-score-game luck is probably the difference.")
    if prob_strength == "narrow":
        reasons.append(f"The edge is slim ({prob_gap:.0f}pp, ~"
                       f"{abs(pred_margin):.0f} pts) — inside one score, "
                       f"where NFL variance (σ≈13.5 pts) dominates.")
    if not reasons:
        reasons.append(f"Offense, defense, and record all point the same way "
                       f"for **{prob_winner}** with a {prob_strength} edge.")
    return level, emoji, color, reasons


# ── GitHub-backed storage (for Streamlit Cloud hosting) ───────────────────────
# Identical mechanism to the MLB/CFB apps — see the MLB file's long comment
# for the full rationale. Short version: Streamlit Cloud wipes local files
# on every reboot, so with [github] secrets configured the log lives in the
# repo on a separate "data" branch. The log FILENAME (nfl_pick_log.json)
# differs from the siblings' so all three apps share one repo/branch
# without clobbering each other.
#
# Setup (once) — same secrets as the other two apps:
#   Streamlit Cloud → app → Settings → Secrets:
#         [github]
#         token  = "github_pat_..."
#         repo   = "your-username/your-repo"
#         branch = "data"

GH_API = "https://api.github.com"
PICK_LOG_FILE = "nfl_pick_log.json"
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
                               "tracker: update nfl pick log")
        st.session_state["_pick_log_cache"] = rows
        st.session_state["_pick_log_sha"] = new_sha
    except Exception as e:
        st.error(f"Pick log NOT saved to GitHub: {e} {_gh_error_hint(e)}")


def flash(msg: str, kind: str = "success"):
    """Queue a status message to show AFTER the next st.rerun(). A plain
    st.success() followed by st.rerun() is wiped before it can be read —
    the tracker buttons all rerun, so their reports go through here."""
    st.session_state["_flash"] = (kind, msg)


def show_flash():
    if "_flash" in st.session_state:
        kind, msg = st.session_state.pop("_flash")
        getattr(st, kind, st.info)(msg)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏈 NFL Weekly Slate")
    st.caption(f"Season: {SEASON} · ESPN data · refreshes every 30 min")

    st.markdown("---")
    st.markdown("##### Settings")
    fade_games = st.select_slider(
        "Prior-season fade (games to fully current)",
        options=[6, 8, 10], value=8,
        help=f"At 0 games played the model runs 100% on {PRIOR_SEASON} "
             f"stats, fading linearly to 100% {SEASON} stats by this many "
             f"games. Faster fade = trusts the new roster sooner.")

    st.markdown("---")
    st.markdown("##### Model weights")
    # Defaults mirror the siblings' 40/40/20 reasoning: record is the most
    # downstream, most luck-contaminated signal (NFL one-score games are
    # near coin flips) and is largely redundant with the scoring numbers
    # that produce it. Sliders stay live.
    w_off_raw = st.slider("Offense weight", 0, 100, 40, step=5)
    w_def_raw = st.slider("Defense weight", 0, 100, 40, step=5)
    w_rec_raw = st.slider("Record weight",  0, 100, 20, step=5)
    total_w = (w_off_raw + w_def_raw + w_rec_raw) or 1
    w_off = w_off_raw / total_w
    w_def = w_def_raw / total_w
    w_rec = w_rec_raw / total_w

    st.markdown("---")
    cur_type, cur_week_default = fetch_current_week(SEASON)
    phase = st.radio("Season phase", ["Regular season", "Playoffs"],
                     index=0 if cur_type == 2 else 1, horizontal=True)
    if phase == "Regular season":
        seasontype = 2
        week = int(st.number_input(
            "Week", min_value=1, max_value=REG_WEEKS,
            value=int(cur_week_default if cur_type == 2 else REG_WEEKS),
            step=1))
        week_label = f"Week {week}"
        # Stats scan: regular-season weeks 1..week (only COMPLETED games
        # count, so the current week's already-played games join the
        # sample while its upcoming games do not).
        reg_upto, post_upto = week, 0
    else:
        seasontype = 3
        rounds = list(POST_ROUNDS)
        week = int(st.selectbox(
            "Round", options=rounds,
            format_func=lambda r: POST_ROUNDS[r],
            index=(rounds.index(cur_week_default)
                   if cur_type == 3 and cur_week_default in rounds else 0)))
        week_label = POST_ROUNDS[week]
        # Full regular season plus every completed earlier round.
        reg_upto, post_upto = REG_WEEKS, week - 1

    with st.spinner(f"Loading {SEASON} results..."):
        cur_stats = fetch_season_team_stats(SEASON, upto_week=reg_upto,
                                            post_upto=post_upto)
    with st.spinner(f"Loading {PRIOR_SEASON} baselines..."):
        prior_stats = fetch_season_team_stats(PRIOR_SEASON,
                                              upto_week=REG_WEEKS, post_upto=0)

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
    st.caption(f"{PRIOR_SEASON} baselines: {len(prior_stats)} teams "
               f"(regular season, opponent-adjusted)")


# ── Main view ──────────────────────────────────────────────────────────────────
(tab_week,) = st.tabs(["🏈 This Week's Games"])

with tab_week:
    st.header(f"{week_label} Slate — {SEASON}")

    with st.spinner("Loading slate..."):
        try:
            slate = fetch_week_slate(SEASON, week, seasontype)
        except Exception as e:
            st.error(f"Couldn't load the slate from ESPN: {e}")
            slate = []

    if not slate:
        st.info("No games found for this week yet.")
        st.stop()

    # Floor rating for a team with no data in either season — should never
    # happen with 32 stable franchises, but if ESPN returns a team ID we
    # have no games for, the card still renders and is flagged as a
    # no-read rather than silently rated. z ≈ -2.5 on everything.
    NO_DATA_PLACEHOLDER = {"pf_pg": 11.0, "pa_pg": 34.0, "mpg": -23.0,
                           "wpct": 10.0, "wins": 0, "losses": 0, "ties": 0,
                           "games": 0}

    slate_results = []
    for g in slate:
        sh_blend, wp_h = blend_prior(cur_stats.get(g["home_id"]),
                                     prior_stats.get(g["home_id"]), fade_games)
        sa_blend, wp_a = blend_prior(cur_stats.get(g["away_id"]),
                                     prior_stats.get(g["away_id"]), fade_games)
        readable = True
        if sh_blend is None:
            sh_blend, readable = dict(NO_DATA_PLACEHOLDER), False
        if sa_blend is None:
            sa_blend, readable = dict(NO_DATA_PLACEHOLDER), False
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
            pred_margin, readable)

        slate_results.append({
            **g,
            "sh": sh_blend, "sa": sa_blend, "w_prior": w_prior,
            "readable": readable,
            "pred_margin": pred_margin,
            "model_line_home": model_line_home,
            "home_pct": pct_h, "away_pct": pct_a,
            "proj_home": proj_h, "proj_away": proj_a, "proj_total": proj_total,
            "prob_pick": pick,
            "conf_level": level, "conf_emoji": emoji,
            "conf_color": color, "conf_reasons": reasons,
        })

    # ── Price gate — all price entry in ONE place, one Enter to apply ─────
    # Same design as the CFB app: two markets per game.
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
            st.caption("For each game, enter YOUR book's numbers: the "
                       "HOME spread (book convention, negative = home "
                       "favored) AND its juice (e.g. -110), plus the ML "
                       "price on the model's pick. Then Apply once. A "
                       "market only counts as priced when its fields are "
                       "explicitly filled — everything left at 0 is "
                       "skipped. Nothing is ever prefilled. Pick'em games: "
                       "the spread field can't tell PK from blank, so take "
                       "the ML (equivalent at PK) instead.")
            for g in upcoming:
                if not g["readable"]:
                    continue
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
                # Deliberately NOT prefilled from ESPN's carried odds:
                # the gate must judge the number at the user's own book.
                # ESPN's line serves one passive role — the CLV reference
                # captured at grading time.
                with fc1:
                    st.number_input("home spread", value=0.0, step=0.5,
                                    key=f"nsp_{mkey}", format="%.1f",
                                    help="Market spread for the HOME team")
                with fc2:
                    st.number_input("spread juice", value=0, step=5,
                                    key=f"nspj_{mkey}",
                                    help="Price on the model's side of the "
                                         "spread (usually -110). Must be "
                                         "typed — a game only counts as "
                                         "priced when BOTH spread and "
                                         "juice are entered.")
                with fc3:
                    st.number_input("ML price", value=0, step=5,
                                    key=f"nml_{mkey}",
                                    help="Price on the model's ML pick")
            st.form_submit_button("Apply odds")

        for g in upcoming:
            if not g["readable"]:
                continue
            mkey = matchup_key(g)

            # ML gate — identical math to the siblings.
            ml_price = st.session_state.get(f"nml_{mkey}", 0)
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
            mkt_sp = st.session_state.get(f"nsp_{mkey}", 0.0) or 0.0
            sp_juice = st.session_state.get(f"nspj_{mkey}", 0)
            has_line = abs(mkt_sp) >= 0.25
            if has_line and abs(sp_juice) >= 100:
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

        # Auto-sync on every load: prices entered in the panel flow into
        # logged rows — the panel is the source of truth for the current
        # week's prices, on the current model version. Graded rows and
        # superseded-version rows are never touched.
        _synced = 0
        _upcoming_by_key = {matchup_key(g): g for g in upcoming}
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
                    _r["prob"] = round(_gc["model"] * 100, 1)
                    _r["edge"] = round(_gc["cushion"] * 100, 1)
                    _synced += 1
                elif (_gc is None and REFERENCE_ATS_ROWS
                        and _r.get("odds", 0) == 0):
                    _g = _upcoming_by_key.get(_r["matchup"])
                    _sp = ((_g.get("odds") or {}).get("home_spread")
                           if _g and _g["state"] == "pre" else None)
                    if _sp is not None:
                        _sp = float(_sp)
                        _cph = cover_prob_home(_g["pred_margin"], _sp)
                        if _cph >= 0.5:
                            _side, _cp, _ln = _g["home"], _cph, _sp
                        else:
                            _side, _cp, _ln = _g["away"], 1 - _cph, -_sp
                        _pick = f"{_side} {fmt_spread(_ln)}"
                        if (_r.get("line") != _ln
                                or _r.get("pick") != _pick):
                            _r["line"] = _ln
                            _r["pick"] = _pick
                            _r["prob"] = round(_cp * 100, 1)
                            _synced += 1
        if _synced:
            save_pick_log(_log)
            st.caption(f"🔄 Updated {_synced} tracker row(s) from the "
                       f"price panel.")

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

    # ── Render each game ───────────────────────────────────────────────────
    for game in slate_results:
        color = game["conf_color"]
        mkey = matchup_key(game)

        if game["completed"]:
            hs, as_ = game["home_score"], game["away_score"]
            if hs == as_:
                final_txt = f"Final · Tie {hs}-{as_}"
            else:
                winner = game["home"] if hs > as_ else game["away"]
                final_txt = f"Final · {winner} won {max(hs,as_)}-{min(hs,as_)}"
            status_badge = (
                f"<span style='background:rgba(0,192,122,0.15);color:#00c07a;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>"
                f"{final_txt}</span>")
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

        if game["intl"]:
            neutral_badge = (f"<span style='font-size:11px;color:#aaa;"
                             f"margin-left:6px;'>🌍 International — "
                             f"{game['venue'] or 'neutral venue'}, no HFA "
                             f"applied</span>")
        elif game["neutral"]:
            neutral_badge = ("<span style='font-size:11px;color:#aaa;"
                             "margin-left:6px;'>🏟 Neutral site — no HFA "
                             "applied</span>")
        else:
            neutral_badge = ""

        sh, sa = game["sh"], game["sa"]
        wp = game["w_prior"]
        if not game["readable"]:
            src_tag = "⚠️ no data for a team"
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

        rec_h = game["home_record"] or fmt_record(sh)
        rec_a = game["away_record"] or fmt_record(sa)

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
        if not game["completed"] and not game["readable"]:
            st.markdown(
                "<div style='font-size:11px;color:#888;"
                "margin:-6px 0 14px 2px;'>No model read for this game — "
                "no prices collected, no gate verdict, and it will not "
                "enter the tracker. A verdict from a placeholder rating "
                "would be fabricated, not conservative.</div>",
                unsafe_allow_html=True)
        elif not game["completed"]:
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
    # Same instrument as the MLB/CFB trackers: logs the live model's picks
    # exactly as displayed (no look-ahead possible), tagged with
    # MODEL_VERSION, every game on the slate — the un-bet games are the
    # control group. ML and ATS are separate rows and are summarized
    # separately: the model can be good at one and bad at the other, and
    # only per-market samples can tell.
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📌 Pick Tracker")
    show_flash()
    _cfg = _gh_cfg()
    if _cfg:
        st.caption(f"🗄 Log storage: GitHub — {_cfg['repo']} @ {_cfg['branch']} "
                   f"(survives Streamlit Cloud reboots)")
    else:
        st.caption("🗄 Log storage: local file — fine on your machine, but "
                   "WIPED on Streamlit Cloud reboots. Add [github] secrets "
                   "to persist (see comment above _gh_cfg in the code).")

    log = load_pick_log()
    if resolve_ats_rows(log):
        save_pick_log(log)
    existing_keys = {(r["date"], r["matchup"], r.get("market", "ML"))
                     for r in log}

    c_log1, c_log3, c_logr, c_log2 = st.columns([1, 1, 1, 1.1])
    with c_log1:
        if st.button("Log this week's slate to tracker"):
            added = 0
            skipped_final = 0
            # Supersede stale versions: UNGRADED rows from an older model
            # version, for games still upcoming on this slate, are removed
            # so the current frozen model can log its own read. GRADED
            # rows are immutable forever, whatever version produced them.
            slate_dm = {((g["kick_date"] or today_et()), matchup_key(g))
                        for g in slate_results if not g["completed"]}
            before = len(log)
            log[:] = [r for r in log if not (
                r.get("version") != MODEL_VERSION
                and not r.get("result")
                and (r["date"], r["matchup"]) in slate_dm)]
            replaced = before - len(log)
            purged_refs = 0
            if not REFERENCE_ATS_ROWS:
                b2 = len(log)
                log[:] = [r for r in log if not (
                    r.get("market") == "ATS" and r.get("odds", 0) == 0
                    and not r.get("result") and "margin" not in r)]
                purged_refs = b2 - len(log)
            existing_keys = {(r["date"], r["matchup"], r.get("market", "ML"))
                             for r in log}
            ats_priced = ats_ref = ats_no_line = 0
            for g in slate_results:
                # Already-final games are EXCLUDED on purpose: a pick logged
                # after the result exists is look-ahead. The forward
                # record only admits games still biddable at log time —
                # but the skip is announced, not silent.
                if g["completed"]:
                    skipped_final += 1
                    continue
                if not g["readable"]:
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
                # ATS rows: priced via the gate, or pending the user's line
                # (model margin pre-registered so the side resolves
                # mechanically once a HOME spread is typed in).
                ga = gate_ats.get(matchup)
                akey = (row_date, matchup, "ATS")
                if ga and akey not in existing_keys:
                    log.append({
                        "date": row_date, "matchup": matchup, "market": "ATS",
                        "pick": ga["pick"],
                        "line": ga["line"],
                        "prob": round(ga["model"] * 100, 1),
                        "tier": g["conf_level"], "version": MODEL_VERSION,
                        "odds": ga["odds"],
                        "edge": round(ga["cushion"] * 100, 1),
                        "margin": round(g["pred_margin"], 1),
                        "result": "",
                    })
                    added += 1
                    ats_priced += 1
                elif ga is None and REFERENCE_ATS_ROWS \
                        and akey not in existing_keys \
                        and (g.get("odds") or {}).get("home_spread") is not None:
                    espn_sp = float(g["odds"]["home_spread"])
                    cp_home = cover_prob_home(g["pred_margin"], espn_sp)
                    if cp_home >= 0.5:
                        side, cp, ln = g["home"], cp_home, espn_sp
                    else:
                        side, cp, ln = g["away"], 1 - cp_home, -espn_sp
                    log.append({
                        "date": row_date, "matchup": matchup,
                        "market": "ATS",
                        "pick": f"{side} {fmt_spread(ln)}",
                        "line": ln,
                        "prob": round(cp * 100, 1),
                        "tier": g["conf_level"],
                        "version": MODEL_VERSION,
                        "odds": 0, "edge": None,
                        "margin": round(g["pred_margin"], 1),
                        "result": "",
                    })
                    added += 1
                    ats_ref += 1
                elif ga is None and akey not in existing_keys:
                    log.append({
                        "date": row_date, "matchup": matchup,
                        "market": "ATS",
                        "pick": PENDING_ATS_PICK,
                        "line": None, "prob": None,
                        "tier": g["conf_level"],
                        "version": MODEL_VERSION,
                        "odds": 0, "edge": None,
                        "margin": round(g["pred_margin"], 1),
                        "result": "",
                    })
                    added += 1
                    ats_no_line += 1
            save_pick_log(log)
            msg = (f"Logged {added} new pick rows."
                   if added else "This week's slate is already logged — "
                                 "prices entered above sync into it "
                                 "automatically.")
            if replaced:
                msg += (f" Replaced {replaced} ungraded row(s) from a "
                        f"superseded model version with the current "
                        f"model's reads.")
            if skipped_final:
                msg += (f" Skipped {skipped_final} already-final game(s): "
                        f"picks can't be logged after the result exists "
                        f"(look-ahead) — only games still biddable at log "
                        f"time enter the forward record.")
            if REFERENCE_ATS_ROWS:
                msg += (f" ATS accounting: {ats_priced} with your prices, "
                        f"{ats_ref} reference-line control rows (odds 0 — "
                        f"graded for calibration, never counted as bets), "
                        f"{ats_no_line} game(s) with no line available.")
            else:
                msg += (f" ATS rows: {ats_priced} priced via the gate, "
                        f"{ats_no_line} pending your line — type each "
                        f"game's HOME spread into the tracker's line "
                        f"column and Save; the model's side resolves "
                        f"automatically from its pre-registered margin.")
                if purged_refs:
                    msg += (f" Purged {purged_refs} leftover reference-line "
                            f"row(s).")
            flash(msg)
            st.rerun()
    with c_log3:
        # Auto-grading pulls finals from the same ESPN API as the slate.
        # ML: winner vs pick; a TIE is a Push (the book refunds). ATS:
        # final margin vs the LOGGED line — margin exactly on the number →
        # Push. Postponed/canceled → Push. CLV captured for both markets
        # from ESPN's last carried odds, best-effort.
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
                # Late kickoffs that cross midnight ET (or the Melbourne-
                # style international slots) can be filed under the
                # neighboring calendar day on ESPN's scoreboard — check
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
                    close_odds = gm.get("odds") or {}
                    if r.get("market", "ML") == "ML":
                        if hs == as_:
                            r["result"] = "Push"
                        else:
                            winner = home if hs > as_ else away
                            r["result"] = "W" if winner == r["pick"] else "L"
                        # ML CLV in probability points: break-even of the
                        # CLOSING price minus break-even of the LOGGED
                        # price, pick side. Positive = the logged price
                        # was better than where the market closed.
                        close_ml = (close_odds.get("home_ml") if r["pick"] == home
                                    else close_odds.get("away_ml"))
                        logged = r.get("odds", 0) or 0
                        try:
                            close_ml = float(close_ml) if close_ml is not None else None
                        except (TypeError, ValueError):
                            close_ml = None
                        if (close_ml is not None and abs(close_ml) >= 100
                                and abs(logged) >= 100):
                            r["close"] = close_ml
                            r["clv"] = round((breakeven_prob(close_ml)
                                              - breakeven_prob(logged)) * 100, 1)
                        graded += 1
                    else:   # ATS
                        line = r.get("line")
                        if line is None:
                            continue
                        side = r["pick"].rsplit(" ", 1)[0]
                        side_margin = (hs - as_) if side == home else (as_ - hs)
                        adj = side_margin + float(line)
                        r["result"] = ("Push" if abs(adj) < 1e-9
                                       else "W" if adj > 0 else "L")
                        # ATS CLV in points: logged line − ESPN's closing
                        # line, pick-side view. Positive = beat the close.
                        close_home = close_odds.get("home_spread")
                        if close_home is not None:
                            try:
                                close_side = (float(close_home) if side == home
                                              else -float(close_home))
                                r["close"] = close_side
                                r["clv"] = round(float(line) - close_side, 1)
                            except (TypeError, ValueError):
                                pass
                        graded += 1
                elif "postponed" in gm["status_detail"].lower() or \
                        "cancel" in gm["status_detail"].lower():
                    r["result"] = "Push"
                    graded += 1
            if graded:
                save_pick_log(log)
                flash(f"Auto-graded {graded} picks.")
                st.rerun()
            else:
                st.info("Nothing new to grade yet.")
    with c_logr:
        # Decontamination: strips every UNGRADED ATS row of the current
        # model version back to pending — line, price, and edge cleared,
        # pre-registered margin kept. Graded rows are untouchable.
        if st.button("🧹 Reset ungraded ATS lines"):
            n_reset = n_dropped = 0
            keep = []
            for r in log:
                if (r.get("market") == "ATS" and not r.get("result")
                        and r.get("version") == MODEL_VERSION):
                    if r.get("margin") is not None:
                        r["pick"] = PENDING_ATS_PICK
                        r["line"] = None
                        r["prob"] = None
                        r["odds"] = 0
                        r["edge"] = None
                        n_reset += 1
                        keep.append(r)
                    else:
                        n_dropped += 1
                else:
                    keep.append(r)
            save_pick_log(keep)
            flash(f"Reset {n_reset} ATS row(s) to pending, removed "
                  f"{n_dropped} legacy row(s) (re-log to recreate "
                  f"them as pending). Type your lines into the "
                  f"tracker and Save.")
            st.rerun()
    with c_log2:
        st.caption("Weekly loop: enter prices in the panel, log the slate "
                   "(2 rows per game — ML and ATS), auto-grade after "
                   "Monday night. Every game gets graded — bet or not. "
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
                    "line", help="ATS rows only. For a pending row, type "
                    "the HOME spread (negative = home favored); on Save "
                    "the model's side resolves and the line restates for "
                    "the pick side. Grading uses THIS number.", step=0.5),
                "margin":  st.column_config.NumberColumn(
                    "margin", help="Model's projected HOME margin, "
                    "pre-registered at log time — drives the ATS side "
                    "once a line exists", disabled=True),
                "prob":    st.column_config.NumberColumn(disabled=True),
                "tier":    st.column_config.TextColumn(disabled=True),
                "version": st.column_config.TextColumn(disabled=True),
                "edge":    st.column_config.NumberColumn(
                    "edge", help="Cushion: model prob − break-even prob of "
                    "the logged price (pp)", disabled=True),
                "close":   st.column_config.NumberColumn(
                    "close", help="ESPN's last carried number for the pick "
                    "side at grading time: spread (ATS) or ML price (ML)",
                    disabled=True),
                "clv":     st.column_config.NumberColumn(
                    "clv", help="Closing line value. ATS: points (logged "
                    "line − close). ML: probability points (close "
                    "break-even − logged break-even). Positive = beat "
                    "the close.", disabled=True),
            },
            num_rows="dynamic",
            hide_index=True, width="stretch", height=300)
        if st.button("Save grades"):
            rows_out = edited.to_dict("records")
            for r in rows_out:
                for k, v in list(r.items()):
                    if isinstance(v, float) and math.isnan(v):
                        r[k] = None
            n_res = resolve_ats_rows(rows_out)
            save_pick_log(rows_out)
            flash(f"Saved. Resolved {n_res} ATS pick(s) from newly "
                  f"entered lines." if n_res else "Saved.")
            st.rerun()

        # Summary — current model version only, graded picks only,
        # SPLIT BY MARKET first: ML skill and ATS skill are different
        # claims and get different verdicts. Pushes are excluded from
        # hit rate and units (stake returned) but are counted so a tie-
        # heavy or key-number-heavy stretch is visible.
        cur = edited[(edited["version"] == MODEL_VERSION) &
                     (edited["result"].isin(["W", "L"]))]
        n_push = int(((edited["version"] == MODEL_VERSION) &
                      (edited["result"] == "Push")).sum())
        if len(cur):
            st.markdown(f"**{MODEL_VERSION}** — graded picks: {len(cur)}"
                        + (f" (+{n_push} push)" if n_push else ""))
            sum_rows = []
            for market in ["ML", "ATS"]:
                m = cur[cur["market"] == market]
                for tier in ["High", "Moderate", "Low"]:
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
                                "points at SIGMA_COVER and the key-number "
                                "problem).")
                    st.dataframe(pd.DataFrame(brows), hide_index=True,
                                 width="stretch")

            # Closing line value — the fastest-converging edge evidence.
            # Includes pushes: a push still had a closing line to beat.
            all_cur = edited[(edited["version"] == MODEL_VERSION) &
                             (edited["result"].isin(["W", "L", "Push"]))]
            if "clv" in all_cur.columns:
                clv_msgs = []
                ats_clv = all_cur[(all_cur["market"] == "ATS")
                                  & all_cur["clv"].notna()]
                if len(ats_clv):
                    good = ats_clv[ats_clv["edge"].fillna(-99) >= 3.0]
                    m = (f"**CLV — ATS:** {ats_clv['clv'].mean():+.2f} "
                         f"pts/pick over {len(ats_clv)}")
                    if len(good):
                        m += (f" · GOOD-bucket: {good['clv'].mean():+.2f} "
                              f"over {len(good)}")
                    clv_msgs.append(m)
                ml_clv = all_cur[(all_cur["market"] == "ML")
                                 & all_cur["clv"].notna()]
                if len(ml_clv):
                    good = ml_clv[ml_clv["edge"].fillna(-99) >= 3.0]
                    m = (f"**CLV — ML:** {ml_clv['clv'].mean():+.2f} "
                         f"pp/pick over {len(ml_clv)}")
                    if len(good):
                        m += (f" · GOOD-bucket: {good['clv'].mean():+.2f} "
                              f"over {len(good)}")
                    clv_msgs.append(m)
                if clv_msgs:
                    st.markdown("<br>".join(clv_msgs) +
                                "<br><span style='font-size:12px;color:#888;'>"
                                "Sustained positive CLV is edge evidence that "
                                "converges far faster than W/L; sustained "
                                "negative CLV on GOOD picks is an early "
                                "warning the model is chasing stale numbers. "
                                "In the NFL, beating the close consistently "
                                "is the whole game.</span>",
                                unsafe_allow_html=True)
            st.caption("Units use only picks with a price entered, flat 1u. "
                       "ATS hit rate must clear ~52.4% at -110 just to break "
                       "even — treat anything under 55% over a small sample "
                       "as noise.")
    else:
        st.caption("No picks logged yet. Log this week's slate to start "
                   "the record.")

"""
NBA Daily Slate — Market Gate & Paper Model
--------------------------------------------
Fourth sibling of the MLB / CFB / NFL apps. Derived from mlb_app.py (v4.2
market gate) for the daily-slate plumbing — odds prefill, closing-line cron
on the data branch, GitHub-backed pick log, flash messages — with the
margin model from nfl_app.py and the SRS-style opponent adjustment from
cfb_app.py. Design contract: NBA_APP_SPEC.md (what gets built). Process:
CLAUDE.md (how work is done). Where they conflict, CLAUDE.md wins on
process and the spec wins on design.

Install dependencies:
    pip install streamlit pandas numpy requests

Run:
    streamlit run nba_app.py

What produces a real bet: THE MARKET GATE ONLY, from opening night —
FanDuel's price vs the sharp reference (Pinnacle, else consensus), spread
and moneyline both, ≥ 1.5pp. The model runs on every game as calibrated
winner-confidence context, and its model-vs-price picks are logged as
PAPER — never bet in v1.0. Promotion is pre-registered (spec §7,
nba_evaluation_plan.md): the earlier of 200 paper picks per bucket or
Feb 19, 2027, looked at ONCE; CLV is the verdict. Pass → the model gate is
ADDED beside the market gate at half stake. Never the model alone.

Data: ESPN's public NBA API (free, no key) for the slate, scores, and the
season-to-date game log — one date-range pull per month (the WNBA app's
pattern), never per-team schedule calls. The Odds API (free tier, 500
credits/month, target ≤ 350) for FanDuel + reference prices.

Test seam: the AppTest harness (tests/test_nba_app.py) pins the clock via
the NBA_APP_CLOCK env var and patches requests.get with mock ESPN / Odds
feeds. Neither is set in production.
"""

import base64
import datetime
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
from zoneinfo import ZoneInfo

from nba_teams import NBA_TEAMS, team_id_for, team_name, team_abbr

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Daily Slate", page_icon="🏀", layout="wide")

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
# Same lesson as the MLB app: Streamlit Cloud runs on UTC, so a naive
# datetime.today() rolls to "tomorrow" at 8 PM Eastern — every evening
# session would fetch and log the NEXT day's slate. Every date, cache key,
# log row, and in-play decision in this app goes through these helpers.
# Never call datetime.datetime.today()/now() directly anywhere below.
APP_TZ = ZoneInfo("America/New_York")
UTC = datetime.timezone.utc


def now_et() -> datetime.datetime:
    # Test seam (AppTest harness only): NBA_APP_CLOCK pins the clock to an
    # ISO datetime so the mock slate is deterministic. Never set in prod.
    pin = os.environ.get("NBA_APP_CLOCK", "").strip()
    if pin:
        dt = datetime.datetime.fromisoformat(pin)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=APP_TZ)
        return dt.astimezone(APP_TZ)
    return datetime.datetime.now(APP_TZ)


def now_utc() -> datetime.datetime:
    return now_et().astimezone(UTC)


def today_et(fmt: str = "%Y-%m-%d") -> str:
    return now_et().strftime(fmt)


def parse_utc(iso: str) -> datetime.datetime | None:
    """ISO-8601 (ESPN 'Z' or Odds API '+00:00') → aware UTC datetime."""
    try:
        dt = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


# ESPN labels an NBA season by the calendar year it ENDS: 2026-27 is
# season=2027, and last season (2025-26, the fallback prior's source) is
# season=2026. From August on, the upcoming season is "the season".
_now = now_et()
SEASON = _now.year + 1 if _now.month >= 8 else _now.year
PRIOR_SEASON = SEASON - 1
SEASON_START_ET = datetime.date(SEASON - 1, 10, 1)   # preseason opens early Oct
REG_SEASON_OPENS = datetime.date(2026, 10, 20)        # market gate live from here
ALL_STAR_BREAK = datetime.date(2027, 2, 19)           # pre-registered look, latest

# ── MODEL FREEZE — nba-v1.0 ────────────────────────────────────────────────────
# Frozen 2026-09-27 (tag nba-v1.0-frozen-<date> before opening night).
# NO PARAMETER CHANGES BEFORE THE PRE-REGISTERED LOOK — see spec §7 and
# nba_evaluation_plan.md: the earlier of 200 `paper_counts` picks in a
# bucket or 2027-02-19, looked at once, CLV the verdict, spread and ML as
# separate buckets. Every logged row carries this string; a new version
# string means a new dataset (never mix). Injuries, 3-in-4, travel, totals
# are OUT of v1.0 by decision (spec §9) — the market gate carries them.
MODEL_VERSION = "nba-v1.0"

# ── Team strength constants (spec §3.1) — hardcoded, no UI controls ───────────
HFA           = 2.5     # points, home (0 on neutral sites — Cup final, intl)
SIGMA         = 12.5    # SD of game margin — ONE distribution for win AND cover
FADE_GAMES    = 15      # prior fades linearly to 0 over this many games
FORM_WINDOW   = 15      # last-N games for recent form
FORM_WEIGHT   = 0.30    # nominal; effective via the double-count correction
B2B_PENALTY   = 1.5     # points, second night of a back-to-back
PRIOR_REGRESS = 0.60    # fallback prior only: prior-season MOV × 0.60
WIN_TOTAL_BASE = 41.0   # prior_margin = (win_total − 41) / 2.7
WIN_TOTAL_PER_PT = 2.7
NBA_PPG_STD   = 4.0     # spread of team scoring; opponent-adjust clamp = 3σ
# Display tiers on the model's pick — context only, the MLB thresholds.
TIER_HIGH_PCT = 65.0
TIER_MOD_PCT  = 56.5
# No K_MARGIN: the CFB look found cover probabilities overclaimed under a
# second scale. SIGMA alone converts margin to both probabilities.

# ── GATES (spec §4) — deliberately NOT adjustable in the UI ───────────────────
# Carried over from the MLB/CFB/NFL apps at Juan's request: no in-app knob,
# so in-the-moment eagerness can't loosen the discipline. Changing these
# requires editing code — that friction is the feature. Any commit that
# changes one must cite the evidence in its message.
MARKET_EDGE_PP   = 1.5   # market gate: FD beats the reference by ≥ this → GOOD
PAPER_CUSHION_PP = 3.0   # paper gate: model beats FD break-even by ≥ this
VERDICT_GOOD = "GOOD PRICE"
VERDICT_THIN = "THIN — NO BET"
VERDICT_AWAY = "STAY AWAY"
VERDICT_ICON = {VERDICT_GOOD: "✅", VERDICT_THIN: "🟡", VERDICT_AWAY: "⛔"}
VERDICT_COLOR = {VERDICT_GOOD: "#00c07a", VERDICT_THIN: "#f5c842",
                 VERDICT_AWAY: "#ff5252"}

# ── Real-money fields (spec §6) — Juan sets these in code before the first
# bet. No UI reload. 0 = not set; the tracker nags and refuses to stake.
STAKE_UNITS   = 0.0     # dollars per position
SEASON_BUDGET = 0.0     # dollars for the whole season

# ── Pre-registered evaluation (spec §7) — recorded here AND in
# nba_evaluation_plan.md. Look ONCE at the earlier of:
EVAL_LOOK_N    = 200            # paper_counts picks in a bucket
EVAL_LOOK_DATE = "2027-02-19"   # All-Star break

# ── Odds feed (The Odds API) ───────────────────────────────────────────────────
# Streamlit Cloud → this app's OWN Settings → Secrets:
#     [odds]
#     api_key   = "..."         # free tier: 500 credits/month, target ≤ 350
#     book      = "fanduel"     # the book you bet at
#     reference = "pinnacle"    # sharp reference; consensus of others if absent
# One pull = markets × 1 = 2 credits (h2h + spreads, bookmakers param set).
# Cached by ET date + hour → 30–60 credits/month from the app.
ODDS_API     = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ODDS_MARKETS = "h2h,spreads"
# Consensus books beside Pinnacle — spec §10 suggests DK / BetMGM / Caesars;
# Juan confirms. Every book here costs nothing extra (cost is per market).
ODDS_BOOKS   = "fanduel,pinnacle,draftkings,betmgm,caesars"

# ── Data branch files (never on main — commits to main reboot the app) ────────
PICK_LOG_FILE   = "nba_pick_log.json"
CLOSES_DIR      = "odds_closes/nba"
WIN_TOTALS_FILE = "nba_preseason_totals_2026.json"   # on main, entered once
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PICK_LOG_PATH = os.path.join(APP_DIR, PICK_LOG_FILE)

# ── ESPN ───────────────────────────────────────────────────────────────────────
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_STANDINGS = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
# Season type from each event's season.type: 1 = preseason, 2 = regular,
# 3 = postseason. Play-in: ESPN's core API lists a separate type 5
# ("Play-In Tournament") — NOT verified against a live feed in the build
# session (egress blocked). SHAKEDOWN ITEM: check dates=20260414 before
# relying on it. Until then any type other than 1/2 is treated as
# postseason (logged, gradeable, excluded from regular-season strength).
SEASON_TYPE_NAMES = {1: "preseason", 2: "regular", 3: "postseason",
                     5: "play-in"}
# NBA Cup final: logged and gradeable, cup_final=True, EXCLUDED from all
# team-strength stats (it doesn't count in standings). Detected from the
# event's notes headline ("NBA Cup ... Championship/Final"); the dated
# neutral-site check is the belt-and-braces fallback.
CUP_FINAL_DATE_ET = "2026-12-11"

# Widget key namespace — fresh and versioned so no session-state carries
# over from a sibling app or an older layout.
NS = "nba1_"


# ── Math helpers ───────────────────────────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |err| < 1.2e-9),
    so the app needs no scipy. Clamped to (1e-9, 1−1e-9)."""
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return ((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))
    q = p - 0.5
    r = q * q
    return ((((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1))


def breakeven_prob(odds: float) -> float:
    """Break-even win probability implied by an American price (includes vig)."""
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def unit_profit(odds: float) -> float:
    """Profit in units on a 1u winning bet at an American price."""
    odds = float(odds)
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def devig(price_a: float, price_b: float) -> float | None:
    """Two-sided devig per book: fair P(a) = BE_a / (BE_a + BE_b)."""
    try:
        if abs(float(price_a)) < 100 or abs(float(price_b)) < 100:
            return None
        ba, bb = breakeven_prob(price_a), breakeven_prob(price_b)
        return ba / (ba + bb)
    except (TypeError, ValueError):
        return None


def is_price(x) -> bool:
    try:
        return x is not None and abs(float(x)) >= 100
    except (TypeError, ValueError):
        return False


def fmt_spread(x) -> str:
    """Display a spread the way books quote it (side view)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if abs(x) < 0.25:
        return "PK"
    return f"{x:+g}"


def fmt_price(x) -> str:
    return f"{int(x):+d}" if is_price(x) else "—"


def _int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


# ── ESPN fetch + parse ─────────────────────────────────────────────────────────
def _espn_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _parse_event(ev: dict) -> dict | None:
    """Flatten one ESPN scoreboard event into the fields we use. Identity is
    the ESPN EVENT ID; teams are ESPN TEAM IDs (spec §1)."""
    try:
        comp = ev["competitions"][0]
        home = away = None
        for c in comp.get("competitors", []) or []:
            team = c.get("team", {}) or {}
            tid = str(team.get("id", "") or "")
            side = {"id": tid,
                    "name": team.get("displayName", "") or team_name(tid),
                    "abbr": team.get("abbreviation", "") or team_abbr(tid),
                    "score": _int(c.get("score")),
                    "record": ""}
            for rec in c.get("records", []) or []:
                if rec.get("type") == "total" or rec.get("name") == "overall":
                    side["record"] = rec.get("summary", "") or ""
            if c.get("homeAway") == "home":
                home = side
            else:
                away = side
        if not home or not away or not home["id"] or not away["id"]:
            return None
        status = ev.get("status") or comp.get("status") or {}
        stype = status.get("type", {}) or {}
        sname = str(stype.get("name", "") or "").upper()
        detail = str(stype.get("shortDetail", "") or stype.get("detail", "") or "")
        state = str(stype.get("state", "") or "")
        completed = bool(stype.get("completed", False)) or sname == "STATUS_FINAL"
        postponed = (sname in ("STATUS_POSTPONED", "STATUS_CANCELED",
                               "STATUS_CANCELLED")
                     or "postpone" in detail.lower() or "cancel" in detail.lower())
        period = _int(status.get("period"))
        ot = bool(completed and period > 4)
        season_type = _int((ev.get("season") or {}).get("type"), 2) or 2
        notes = " ".join(str(n.get("headline", "") or "")
                         for n in (comp.get("notes", []) or []))
        nl = notes.lower()
        neutral = bool(comp.get("neutralSite", False))
        tip = parse_utc(ev.get("date", ""))
        if tip is None:
            return None
        tip_et = tip.astimezone(APP_TZ)
        date_et = tip_et.strftime("%Y-%m-%d")
        tip_label = tip_et.strftime("%a %I:%M %p ET").replace(" 0", " ")
        cup_final = (("cup" in nl and ("championship" in nl or "final" in nl))
                     or (date_et == CUP_FINAL_DATE_ET and neutral))
        playin = season_type == 5 or "play-in" in nl or "play in" in nl
        venue = comp.get("venue", {}) or {}
        return {
            "event_id": str(ev.get("id", "")),
            "home_id": home["id"], "home": home["name"], "home_abbr": home["abbr"],
            "home_score": home["score"], "home_record": home["record"],
            "away_id": away["id"], "away": away["name"], "away_abbr": away["abbr"],
            "away_score": away["score"], "away_record": away["record"],
            "season_type": season_type,
            "season_type_name": SEASON_TYPE_NAMES.get(season_type, "postseason"),
            "preseason": season_type == 1,
            "regular": season_type == 2,
            "playin": playin,
            "neutral": neutral, "cup_final": cup_final,
            "completed": completed, "postponed": postponed,
            "state": state, "status_name": sname, "status_detail": detail,
            "period": period, "ot": ot,
            "tip_utc": tip.isoformat(), "tip_et": tip_et.isoformat(),
            "date_et": date_et, "tip_label": tip_label,
            "venue": venue.get("fullName", "") or "",
            "notes": notes,
        }
    except Exception:
        return None


def _month_chunks(start: datetime.date, end: datetime.date) -> list:
    """Split [start, end] into calendar-month-aligned pairs, inclusive."""
    chunks = []
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        chunk_end = min(next_month - datetime.timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + datetime.timedelta(days=1)
    return chunks


def _scoreboard_events_for_day(ymd: str) -> list:
    try:
        return _espn_get(f"{ESPN}/scoreboard", {"dates": ymd, "limit": 100}
                         ).get("events", []) or []
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_season_events(season: int, upto_iso: str) -> list:
    """Every ESPN event this season from Oct 1 through `upto_iso` (the ET
    date — part of the cache key so a cached season can't straddle the day
    boundary). ONE range pull per month (WNBA pattern: dates=YYYYMMDD-
    YYYYMMDD&limit=N), with a concurrent per-day fallback if a month's
    range request errors or comes back empty, so a quirk in the range API
    can never silently lose games. This single scan feeds games played,
    MOV, opponent adjustment, last-15 form, and back-to-back detection —
    no per-team schedule calls anywhere."""
    start = datetime.date(season - 1, 10, 1)
    end = datetime.date.fromisoformat(upto_iso)
    if start > end:
        return []
    events = {}
    for c_start, c_end in _month_chunks(start, end):
        raw = None
        try:
            raw = _espn_get(f"{ESPN}/scoreboard",
                            {"dates": f"{c_start:%Y%m%d}-{c_end:%Y%m%d}",
                             "limit": 1000}).get("events", []) or []
        except Exception:
            raw = None
        if not raw:
            days = [(c_start + datetime.timedelta(days=i)).strftime("%Y%m%d")
                    for i in range((c_end - c_start).days + 1)]
            raw = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_scoreboard_events_for_day, d) for d in days]
                for fut in as_completed(futures):
                    raw.extend(fut.result())
        for ev in raw:
            g = _parse_event(ev)
            if g:
                events[g["event_id"]] = g
    return sorted(events.values(), key=lambda g: g["tip_utc"])


@st.cache_data(show_spinner=False, ttl=600)
def fetch_scoreboard(date_iso: str) -> list:
    """All parsed events ESPN lists for one ET date. date_iso is part of the
    cache key so a cached slate never survives the ET day boundary. Used for
    today's slate (fresh statuses) and for auto-grading."""
    ymd = date_iso.replace("-", "")
    j = _espn_get(f"{ESPN}/scoreboard", {"dates": ymd, "limit": 100})
    games = [_parse_event(ev) for ev in j.get("events", []) or []]
    return [g for g in games if g]


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_prior_mov(prior_season: int) -> dict:
    """{team_id: prior-season MOV} from ESPN standings ?season=<prior>.
    Only needed for the fallback prior (a team missing from the win-totals
    file). Prefers avgPointsFor − avgPointsAgainst (unambiguously per game);
    falls back to a differential stat, dividing by games if it's a total."""
    try:
        j = _espn_get(ESPN_STANDINGS, {"season": prior_season})
    except Exception:
        return {}
    entries = []
    for grp in j.get("children", []) or []:
        entries += (grp.get("standings") or {}).get("entries", []) or []
    if not entries:
        entries = (j.get("standings") or {}).get("entries", []) or []
    out = {}
    for e in entries:
        tid = str((e.get("team") or {}).get("id", "") or "")
        if tid not in NBA_TEAMS:
            continue
        stats = {}
        for s in e.get("stats", []) or []:
            if s.get("name") is not None and s.get("value") is not None:
                stats[s["name"]] = s["value"]
        pf, pa = _float(stats.get("avgPointsFor")), _float(stats.get("avgPointsAgainst"))
        if pf is not None and pa is not None:
            out[tid] = round(pf - pa, 2)
            continue
        diff = _float(stats.get("pointDifferential", stats.get("differential")))
        if diff is None:
            continue
        gp = _float(stats.get("gamesPlayed")) or (
            (_float(stats.get("wins")) or 0) + (_float(stats.get("losses")) or 0))
        out[tid] = round(diff / gp if gp and abs(diff) > 30 else diff, 2)
    return out


def load_win_totals() -> tuple[dict, str]:
    """Preseason win totals entered once by Juan (spec §3.1). Missing file
    or null entries → the fallback prior for those teams."""
    path = os.path.join(APP_DIR, WIN_TOTALS_FILE)
    try:
        with open(path, "r") as f:
            j = json.load(f)
    except Exception:
        return {}, "missing"
    teams = j.get("teams", j) if isinstance(j, dict) else {}
    out = {}
    for tid, v in teams.items():
        if str(tid) not in NBA_TEAMS:
            continue
        wt = v.get("win_total") if isinstance(v, dict) else v
        wt = _float(wt)
        if wt is not None and 0 < wt < 82:
            out[str(tid)] = wt
    return out, "ok"


# ── Team strength (spec §3.1) ──────────────────────────────────────────────────
def build_gamelogs(events: list) -> tuple[dict, set, dict]:
    """From every parsed event this season:
      season_log  {tid: [(opp_id, pf, pa), ...]}  regular-season games only,
                  completed, chronological, EXCLUDING the NBA Cup final
                  (it doesn't count in standings) — feeds MOV/SRS/form;
      played      {(tid, date_et)}  every completed game of ANY type
                  (preseason counts too) — back-to-back detection;
      records     {tid: [w, l]}  regular-season W-L."""
    season_log, played, records = {}, set(), {}
    for g in sorted(events, key=lambda e: e["tip_utc"]):
        if not g["completed"] or g["postponed"]:
            continue
        hid, aid = g["home_id"], g["away_id"]
        played.add((hid, g["date_et"]))
        played.add((aid, g["date_et"]))
        if not g["regular"] or g["cup_final"]:
            continue
        hs, as_ = g["home_score"], g["away_score"]
        if hs == 0 and as_ == 0:
            continue
        season_log.setdefault(hid, []).append((aid, hs, as_))
        season_log.setdefault(aid, []).append((hid, as_, hs))
        rh = records.setdefault(hid, [0, 0])
        ra = records.setdefault(aid, [0, 0])
        if hs > as_:
            rh[0] += 1; ra[1] += 1
        else:
            ra[0] += 1; rh[1] += 1
    return season_log, played, records


def opponent_adjust(gamelog: dict, n_iter: int = 30, damp: float = 0.7) -> dict:
    """
    SRS-style iterative opponent adjustment — the cfb_app.py / nfl_app.py
    block, structure unchanged, on points scored/allowed per game. Each pass
    re-values every team's offense and defense against its opponents'
    CURRENT adjusted ratings:

        adj_off(i) = mean over i's games of (pts scored − adj_def(opp))
                     + league mean
        adj_def(i) = mean over i's games of (pts allowed − adj_off(opp))
                     + league mean          (lower = better defense)

    Damped and re-centered each pass so early-season sparse schedules can't
    oscillate or drift. Clamp band = 3σ of NBA team scoring. Returns
    {tid: {pf_pg, pa_pg, mpg (adjusted), raw_mpg, games, sos}}.
    """
    stats = {}
    for t, games in gamelog.items():
        if not games:
            continue
        n = len(games)
        pf = sum(g[1] for g in games) / n
        pa = sum(g[2] for g in games) / n
        stats[t] = {"pf_pg": pf, "pa_pg": pa, "games": n, "raw_mpg": pf - pa}
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
    band = 3 * NBA_PPG_STD
    for t, s in stats.items():
        s["pf_pg"] = min(mean_off + band, max(mean_off - band, adj_off[t]))
        s["pa_pg"] = min(mean_def + band, max(mean_def - band, adj_def[t]))
        s["mpg"] = s["pf_pg"] - s["pa_pg"]
        s["sos"] = round(s["mpg"] - s["raw_mpg"], 2)
    return stats


def form_weight(n: int) -> float:
    """Effective weight on the last-FORM_WINDOW games, with the double-count
    correction from the MLB lesson: the season average already CONTAINS the
    window, so blending it in again at the nominal 30% would give the last
    15 games 30% + 70%·share. Solve for the internal weight that makes 30%
    the TRUE effective weight:
        share = min(FORM_WINDOW, n) / n
        w_int = max(0, (FORM_WEIGHT − share) / (1 − share))
    → 0 while n ≤ FORM_WINDOW (the window IS the season), and in fact 0
    until share < FORM_WEIGHT (n > 50): before that the season average
    alone already weights the window above 30%."""
    n = int(n)
    if n <= FORM_WINDOW:
        return 0.0
    share = FORM_WINDOW / n
    return max(0.0, (FORM_WEIGHT - share) / (1.0 - share))


def prior_weight(n: int) -> float:
    """Linear fade over FADE_GAMES regular-season games."""
    return max(0.0, 1.0 - int(n) / FADE_GAMES)


def build_priors(win_totals: dict, prior_mov: dict) -> dict:
    """Per-team prior margin before any games (spec §3.1).
    Primary: (win_total − 41) / 2.7 from the totals file.
    Fallback: prior-season MOV × PRIOR_REGRESS from ESPN standings.
    Last resort: 0 (league average), flagged."""
    out = {}
    for tid in NBA_TEAMS:
        if tid in win_totals:
            wt = win_totals[tid]
            out[tid] = {"margin": (wt - WIN_TOTAL_BASE) / WIN_TOTAL_PER_PT,
                        "source": "win total", "detail": f"{wt:.1f} wins"}
        elif tid in prior_mov:
            out[tid] = {"margin": prior_mov[tid] * PRIOR_REGRESS,
                        "source": "prior MOV × 0.60",
                        "detail": f"{prior_mov[tid]:+.1f} MOV in {PRIOR_SEASON - 1}-{PRIOR_SEASON % 100:02d}"}
        else:
            out[tid] = {"margin": 0.0, "source": "none",
                        "detail": "no win total, no prior-season MOV → 0"}
    return out


def team_strengths(season_log: dict, priors: dict) -> dict:
    """{tid: {n, prior, prior_src, season_adj, form_adj, w_form, w_prior,
    blend, strength, raw_mpg, sos}} — points of margin vs an average team."""
    season_adj = opponent_adjust(season_log)
    form_adj = opponent_adjust({t: g[-FORM_WINDOW:] for t, g in season_log.items()})
    out = {}
    for tid in NBA_TEAMS:
        n = len(season_log.get(tid, []))
        s_adj = season_adj.get(tid, {}).get("mpg", 0.0)
        f_adj = form_adj.get(tid, {}).get("mpg", 0.0)
        w_f = form_weight(n)
        blend = (1 - w_f) * s_adj + w_f * f_adj
        w_p = prior_weight(n)
        pr = priors.get(tid, {"margin": 0.0, "source": "none", "detail": ""})
        out[tid] = {
            "n": n, "prior": pr["margin"], "prior_src": pr["source"],
            "prior_detail": pr.get("detail", ""),
            "season_adj": s_adj, "form_adj": f_adj, "w_form": w_f,
            "w_prior": w_p, "blend": blend,
            "strength": w_p * pr["margin"] + (1 - w_p) * blend,
            "raw_mpg": season_adj.get(tid, {}).get("raw_mpg", 0.0),
            "sos": season_adj.get(tid, {}).get("sos", 0.0),
        }
    return out


# ── Game projection (spec §3.2) ────────────────────────────────────────────────
def project_margin(s_home: float, s_away: float, neutral: bool,
                   home_b2b: bool, away_b2b: bool) -> tuple[float, float, float]:
    """Projected HOME margin M = (strength_home − strength_away) + hfa +
    rest_adj, where rest_adj = B2B_PENALTY·(away_on_b2b − home_on_b2b)
    (positive favors home; nets to 0 when both are). Returns (M, hfa, rest)."""
    rest = B2B_PENALTY * (int(bool(away_b2b)) - int(bool(home_b2b)))
    hfa = 0.0 if neutral else HFA
    return (s_home - s_away) + hfa + rest, hfa, rest


def p_home_win(M: float) -> float:
    return norm_cdf(M / SIGMA)


def p_home_cover(M: float, home_spread: float) -> float:
    """P(home covers) at a HOME spread s (negative = home favored). Home must
    win by L = −s: Φ((M − L)/σ) = Φ((M + s)/σ)."""
    return norm_cdf((M + float(home_spread)) / SIGMA)


def tier_for(p_pick: float) -> str:
    pct = p_pick * 100
    return "High" if pct >= TIER_HIGH_PCT else ("Moderate" if pct >= TIER_MOD_PCT else "Low")


# ── Gates (spec §4) ────────────────────────────────────────────────────────────
def verdict(edge: float | None, thresh_pp: float) -> str | None:
    if edge is None:
        return None
    if edge >= thresh_pp / 100.0:
        return VERDICT_GOOD
    if edge >= 0:
        return VERDICT_THIN
    return VERDICT_AWAY


def fair_margin_from_spread(home_spread, home_price, away_price) -> float | None:
    """A book's spread + two-sided prices → its fair HOME margin:
    M* = L + σ·Φ⁻¹(p), L = −home_spread, p = devigged P(home covers)."""
    p = devig(home_price, away_price)
    if p is None or home_spread is None:
        return None
    return -float(home_spread) + SIGMA * norm_ppf(p)


def reference_lines(books: dict, fd_book: str, ref_book: str) -> dict:
    """The sharp reference per market: `ref_book` (Pinnacle) if it quotes
    the market, else the consensus (mean) of the other non-FanDuel books.
    Spread consensus averages each book's fair margin M*; ML consensus
    averages devigged home probabilities. books: {book: {"h2h": {"home":
    px, "away": px}, "spreads": {"home": (pt, px), "away": (pt, px)}}}."""
    def spread_of(q):
        sp = q.get("spreads")
        if not sp:
            return None, None
        (pt_h, px_h), (pt_a, px_a) = sp["home"], sp["away"]
        return fair_margin_from_spread(pt_h, px_h, px_a), sp
    def ml_of(q):
        h2h = q.get("h2h")
        if not h2h:
            return None, None
        return devig(h2h["home"], h2h["away"]), h2h

    ats = {"src": None, "fair_margin": None, "spread": None,
           "sp_home": None, "sp_away": None, "n": 0}
    ml = {"src": None, "fair_home": None, "ml_home": None, "ml_away": None, "n": 0}
    if ref_book in books:
        fm, sp = spread_of(books[ref_book])
        if fm is not None:
            ats.update(src=ref_book, fair_margin=fm, spread=sp["home"][0],
                       sp_home=sp["home"][1], sp_away=sp["away"][1], n=1)
        pm, h2h = ml_of(books[ref_book])
        if pm is not None:
            ml.update(src=ref_book, fair_home=pm, ml_home=h2h["home"],
                      ml_away=h2h["away"], n=1)
    others = {bk: q for bk, q in books.items() if bk not in (fd_book, ref_book)}
    if ats["src"] is None:
        fms = [(fm, sp) for fm, sp in (spread_of(q) for q in others.values())
               if fm is not None]
        if fms:
            ats.update(src=f"consensus of {len(fms)}",
                       fair_margin=float(np.mean([f for f, _ in fms])),
                       spread=float(np.mean([sp["home"][0] for _, sp in fms])),
                       sp_home=int(np.mean([sp["home"][1] for _, sp in fms])),
                       sp_away=int(np.mean([sp["away"][1] for _, sp in fms])),
                       n=len(fms))
    if ml["src"] is None:
        pms = [(pm, h2h) for pm, h2h in (ml_of(q) for q in others.values())
               if pm is not None]
        if pms:
            ml.update(src=f"consensus of {len(pms)}",
                      fair_home=float(np.mean([p for p, _ in pms])),
                      ml_home=int(np.mean([h["home"] for _, h in pms])),
                      ml_away=int(np.mean([h["away"] for _, h in pms])),
                      n=len(pms))
    return {"ats": ats, "ml": ml}


def market_gate(fd: dict, ref: dict) -> dict:
    """Market gate (spec §4.1): is FanDuel's price better than the sharp
    reference's fair line? Both sides, both markets. fd = {"spread",
    "sp_home", "sp_away", "ml_home", "ml_away"} (None where absent).
    Spread: fair_p = Φ((M* − L_F)/σ) for the home side at FD's line
    (M* negated and the line flipped for the away side); edge = fair_p −
    BE(FD price). ML: edge = devigged reference prob − BE(FD price).
    Returns {"ats": {...} | None, "ml": {...} | None}, each with per-side
    dicts and the best side's verdict."""
    out = {"ats": None, "ml": None}
    s = fd.get("spread")
    M = ref["ats"]["fair_margin"]
    if s is not None and M is not None and (is_price(fd.get("sp_home"))
                                            or is_price(fd.get("sp_away"))):
        sides = {}
        for side, m_side, line, price in (("home", M, float(s), fd.get("sp_home")),
                                          ("away", -M, -float(s), fd.get("sp_away"))):
            if not is_price(price):
                continue
            fair = norm_cdf((m_side + line) / SIGMA)     # L_side = −line
            be = breakeven_prob(price)
            edge = fair - be
            sides[side] = {"fair": fair, "be": be, "edge": edge, "line": line,
                           "price": int(price),
                           "verdict": verdict(edge, MARKET_EDGE_PP)}
        if sides:
            best = max(sides, key=lambda k: sides[k]["edge"])
            out["ats"] = {"sides": sides, "side": best,
                          "verdict": sides[best]["verdict"],
                          "edge": sides[best]["edge"], "src": ref["ats"]["src"]}
    ph = ref["ml"]["fair_home"]
    if ph is not None and (is_price(fd.get("ml_home")) or is_price(fd.get("ml_away"))):
        sides = {}
        for side, fair, price in (("home", ph, fd.get("ml_home")),
                                  ("away", 1 - ph, fd.get("ml_away"))):
            if not is_price(price):
                continue
            be = breakeven_prob(price)
            edge = fair - be
            sides[side] = {"fair": fair, "be": be, "edge": edge, "line": None,
                           "price": int(price),
                           "verdict": verdict(edge, MARKET_EDGE_PP)}
        if sides:
            best = max(sides, key=lambda k: sides[k]["edge"])
            out["ml"] = {"sides": sides, "side": best,
                         "verdict": sides[best]["verdict"],
                         "edge": sides[best]["edge"], "src": ref["ml"]["src"]}
    return out


def paper_gate(M: float, fd: dict) -> dict:
    """Paper gate (spec §4.2): the MODEL's pick side vs FanDuel's break-even,
    cushion ≥ PAPER_CUSHION_PP. Logged under paper_* fields, NEVER bet in
    v1.0 — no bet button is wired to these. paper_ml: side with P_win >
    0.5; paper_ats: side with P_cover > 0.5 at FanDuel's line."""
    ph = p_home_win(M)
    ml_side = "home" if ph >= 0.5 else "away"
    p_ml = ph if ml_side == "home" else 1 - ph
    ml_price = fd.get("ml_home") if ml_side == "home" else fd.get("ml_away")
    ml = {"side": ml_side, "p": p_ml, "price": None, "be": None,
          "cushion": None, "verdict": None}
    if is_price(ml_price):
        be = breakeven_prob(ml_price)
        ml.update(price=int(ml_price), be=be, cushion=p_ml - be,
                  verdict=verdict(p_ml - be, PAPER_CUSHION_PP))
    ats = None
    s = fd.get("spread")
    if s is not None:
        pc = p_home_cover(M, s)
        side = "home" if pc >= 0.5 else "away"
        p = pc if side == "home" else 1 - pc
        line = float(s) if side == "home" else -float(s)
        price = fd.get("sp_home") if side == "home" else fd.get("sp_away")
        ats = {"side": side, "p": p, "line": line, "price": None, "be": None,
               "cushion": None, "verdict": None}
        if is_price(price):
            be = breakeven_prob(price)
            ats.update(price=int(price), be=be, cushion=p - be,
                       verdict=verdict(p - be, PAPER_CUSHION_PP))
    return {"ml": ml, "ats": ats}


def position_from_gate(mg: dict) -> dict | None:
    """Position rule (spec §4.3): GOOD PRICE on both the spread and the ML
    of the SAME side of the same game = ONE position — the spread by
    default, ML noted. Never two stakes on one side of one game."""
    ats, ml = mg.get("ats"), mg.get("ml")
    ats_good = bool(ats and ats["verdict"] == VERDICT_GOOD)
    ml_good = bool(ml and ml["verdict"] == VERDICT_GOOD)
    if ats_good:
        sd = ats["sides"][ats["side"]]
        pos = {"market": "ATS", "side": ats["side"], "line": sd["line"],
               "price": sd["price"], "edge": sd["edge"], "note": ""}
        if ml_good and ml["side"] == ats["side"]:
            pos["note"] = (f"ML also GOOD ({ml['edge']*100:+.1f}pp) — one "
                           f"position per side: spread taken, ML noted")
        elif ml_good:
            pos["note"] = (f"ML GOOD on the other side ({ml['edge']*100:+.1f}pp) "
                           f"— one position per game: spread taken")
        return pos
    if ml_good:
        sd = ml["sides"][ml["side"]]
        return {"market": "ML", "side": ml["side"], "line": None,
                "price": sd["price"], "edge": sd["edge"], "note": ""}
    return None


# ── Grading helpers (spec §6) ──────────────────────────────────────────────────
def side_key(row: dict, name) -> str | None:
    """'home' / 'away' for a stored side (a team name or literal key)."""
    if name in ("home", "away"):
        return name
    if not name:
        return None
    tid = team_id_for(name)
    if tid and tid == str(row.get("home_id")):
        return "home"
    if tid and tid == str(row.get("away_id")):
        return "away"
    return None


def grade_ats(side: str, line_side: float, fh: int, fa: int) -> str:
    """W / L / Push for a side at its own line (side view: home −3.5 →
    −3.5, away +3.5 → +3.5). Whole-number line landing exactly → Push:
    stake returned, excluded from W-L, CLV still recorded."""
    margin = (fh - fa) if side == "home" else (fa - fh)
    adj = margin + float(line_side)
    if abs(adj) < 1e-9:
        return "Push"
    return "W" if adj > 0 else "L"


def grade_ml(side: str, fh: int, fa: int) -> str:
    """No ties in the NBA (OT decides). W / L."""
    winner = "home" if fh > fa else "away"
    return "W" if winner == side else "L"


def clv_ats_pts(line_side: float, close_side: float) -> float:
    """ATS CLV in spread points, side view: logged − close. Took home −3.5,
    closed −4.5 → +1.0 (beat the close)."""
    return round(float(line_side) - float(close_side), 1)


def clv_ml_pp(price_side: float, close_side: float) -> float:
    """ML CLV in probability points: BE(close) − BE(taken). Took −110
    (52.4%), closed −120 (54.5%) → +2.1pp (beat the close)."""
    return round((breakeven_prob(close_side) - breakeven_prob(price_side)) * 100, 2)


def units_for(result: str, price) -> float:
    if result == "W" and is_price(price):
        return unit_profit(price)
    if result == "L":
        return -1.0
    return 0.0


# ── Odds feed ──────────────────────────────────────────────────────────────────
def _odds_cfg():
    try:
        o = st.secrets["odds"]
        return {"key": str(o["api_key"]).strip(),
                "book": str(o.get("book", "fanduel")).strip().lower(),
                "ref": str(o.get("reference", "pinnacle")).strip().lower()}
    except Exception:
        return None


def parse_odds_event(ev: dict) -> dict | None:
    """One Odds API event → {home_id, away_id, commence, books}. Team joins
    go through nba_teams.team_id_for (never raw names)."""
    hid, aid = team_id_for(ev.get("home_team")), team_id_for(ev.get("away_team"))
    if not hid or not aid:
        return None
    books = {}
    for bk in ev.get("bookmakers", []) or []:
        q = {}
        for mk in bk.get("markets", []) or []:
            outs = mk.get("outcomes", []) or []
            if mk.get("key") == "h2h":
                d = {team_id_for(o.get("name")): _int(o.get("price")) for o in outs}
                if hid in d and aid in d:
                    q["h2h"] = {"home": d[hid], "away": d[aid]}
            elif mk.get("key") == "spreads":
                d = {}
                for o in outs:
                    pt = _float(o.get("point"))
                    if pt is not None:
                        d[team_id_for(o.get("name"))] = (pt, _int(o.get("price")))
                if hid in d and aid in d:
                    q["spreads"] = {"home": d[hid], "away": d[aid]}
        if q:
            books[bk.get("key", "")] = q
    return {"home_id": hid, "away_id": aid,
            "commence": ev.get("commence_time", ""), "books": books}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_market_odds(api_key: str, cache_key: str) -> tuple:
    """Current h2h + spreads from the books in ODDS_BOOKS — ONE pull, 2
    credits. cache_key = ET date + hour (spec §5), so at most one pull per
    hour of use. Returns (events, remaining_credits, error)."""
    try:
        r = requests.get(ODDS_API, params={
            "apiKey": api_key, "markets": ODDS_MARKETS, "oddsFormat": "american",
            "bookmakers": ODDS_BOOKS}, timeout=20)
        remaining = r.headers.get("x-requests-remaining") if hasattr(r, "headers") else None
        if r.status_code != 200:
            return [], remaining, f"odds API {r.status_code}: {str(r.text)[:120]}"
        out = [parse_odds_event(ev) for ev in r.json()]
        return [e for e in out if e], remaining, ""
    except Exception as e:
        return [], None, f"odds fetch failed: {e}"


def match_odds_event(events: list, g: dict) -> tuple:
    """Match a slate game to a feed event by ESPN team IDs and tip time.
    PRE-GAME quotes only: the feed also returns LIVE in-play prices for
    games underway — meaningless for a pre-game decision, skipped."""
    now = now_utc()
    tip = parse_utc(g["tip_utc"])
    same = [e for e in events
            if e["home_id"] == g["home_id"] and e["away_id"] == g["away_id"]]
    cands = []
    for e in same:
        c = parse_utc(e["commence"])
        if c is None:
            continue
        if tip is not None and abs((c - tip).total_seconds()) > 6 * 3600:
            continue
        cands.append((c, e))
    if not cands:
        return None, "unmatched"
    upcoming = sorted([(c, e) for c, e in cands if c > now], key=lambda x: x[0])
    if not upcoming:
        return None, "live"
    return upcoming[0][1], "ok"


# ── GitHub-backed storage (for Streamlit Cloud hosting) ───────────────────────
# Identical mechanism to the MLB/CFB/NFL apps — see mlb_app.py's long
# comment for the full rationale. Short version: Streamlit Cloud wipes
# local files on every reboot, so with [github] secrets configured the
# log lives in the repo on a separate "data" branch (commits to main
# reboot the app). The log FILENAME (nba_pick_log.json) and the closes
# directory (odds_closes/nba/) are this app's own, so all four apps share
# one repo/branch without clobbering each other.
#
# Setup (once) — this app's OWN Streamlit Secrets (spec §1):
#   Streamlit Cloud → app → Settings → Secrets:
#         [github]
#         token  = "github_pat_..."     # fine-grained, Contents: read/write, this repo only
#         repo   = "etheriousleo/MLB-Prediction-Model"
#         branch = "data"
# Never commit the token. Without secrets, everything falls back to local files.

GH_API = "https://api.github.com"


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
                               "tracker: update nba pick log")
        st.session_state["_pick_log_cache"] = rows
        st.session_state["_pick_log_sha"] = new_sha
    except Exception as e:
        st.error(f"Pick log NOT saved to GitHub: {e} {_gh_error_hint(e)}")


def load_closes(date_iso: str) -> dict:
    """Closing lines captured by scripts/snapshot_closes_nba.py:
    odds_closes/nba/<ET date>.json on the data branch, keyed by ESPN event
    ID. Local file fallback when no [github] secrets (dev / harness).
    Session-cached per date."""
    key = f"_closes_{date_iso}"
    if key in st.session_state:
        return st.session_state[key]
    snap = {}
    cfg = _gh_cfg()
    try:
        if cfg:
            snap, _ = _gh_get_file(cfg, f"{CLOSES_DIR}/{date_iso}.json")
        else:
            p = os.path.join(APP_DIR, CLOSES_DIR, f"{date_iso}.json")
            if os.path.exists(p):
                with open(p, "r") as f:
                    snap = json.load(f)
    except Exception:
        snap = {}
    snap = snap if isinstance(snap, dict) else {}
    st.session_state[key] = snap
    return snap


def flash(msg: str, kind: str = "success"):
    """Queue a status message to show AFTER the next st.rerun(). A plain
    st.success() followed by st.rerun() is wiped before it can be read —
    the tracker buttons all rerun, so their reports go through here."""
    st.session_state["_flash"] = (kind, msg)


def show_flash():
    if "_flash" in st.session_state:
        kind, msg = st.session_state.pop("_flash")
        getattr(st, kind, st.info)(msg)


# ── Tracker row (spec §6) ──────────────────────────────────────────────────────
def side_name(g: dict, side) -> str | None:
    if side == "home":
        return g["home"]
    if side == "away":
        return g["away"]
    return None


def pct(p) -> float | None:
    return None if p is None else round(float(p) * 100, 1)


def pp(x) -> float | None:
    return None if x is None else round(float(x) * 100, 2)


def build_row(g: dict, fd: dict, ref: dict, mg: dict, pg: dict,
              pos: dict | None) -> dict:
    """One tracker row per game — the MLB row's shared fields (date,
    matchup, pick, prob, tier, version, odds, edge, stake, result) keep
    their names; everything NBA-specific is added. Sides are stored as team
    names (readable in the tracker); grading resolves them via side_key."""
    ats, ml = mg.get("ats"), mg.get("ml")
    pats, pml = pg.get("ats"), pg.get("ml")
    ats_sides = ats["sides"] if ats else {}
    ml_sides = ml["sides"] if ml else {}
    row = {
        # shared MLB names
        "date": g["date_et"], "matchup": matchup_key(g),
        "pick": side_name(g, pml["side"]) if pml else None,
        "prob": pct(pml["p"]) if pml else None,
        "tier": g["conf_level"], "version": MODEL_VERSION,
        "odds": pos["price"] if pos else 0,
        "edge": pp(pos["edge"]) if pos else None,
        "stake": 0.0, "result": "",
        # identity
        "espn_event_id": g["event_id"], "date_et": g["date_et"],
        "tip_et": g["tip_et"], "home_id": g["home_id"], "away_id": g["away_id"],
        "home": g["home"], "away": g["away"],
        "season_type": g["season_type"], "neutral_site": bool(g["neutral"]),
        "cup_final": bool(g["cup_final"]),
        "home_b2b": bool(g["home_b2b"]), "away_b2b": bool(g["away_b2b"]),
        # model
        "model_version": MODEL_VERSION,
        "model_margin": round(g["pred_margin"], 2),
        "hfa": g["hfa"], "rest_adj": g["rest"],
        "p_home_win": pct(g["p_home"]),
        "w_prior_home": round(g["sh"]["w_prior"], 3),
        "w_prior_away": round(g["sa"]["w_prior"], 3),
        # FanDuel
        "fd_source": fd.get("source", "none"),
        "fd_spread": fd.get("spread"), "fd_spread_price": fd.get("sp_home"),
        "fd_spread_price_away": fd.get("sp_away"),
        "fd_ml_home": fd.get("ml_home"), "fd_ml_away": fd.get("ml_away"),
        # reference
        "ref_book": ref["ats"]["src"] or ref["ml"]["src"],
        "ref_spread": ref["ats"]["spread"], "ref_spread_price": ref["ats"]["sp_home"],
        "ref_spread_price_away": ref["ats"]["sp_away"],
        "ref_ml_home": ref["ml"]["ml_home"], "ref_ml_away": ref["ml"]["ml_away"],
        "fair_margin": (None if ref["ats"]["fair_margin"] is None
                        else round(ref["ats"]["fair_margin"], 2)),
        "ref_fair_ml_home": pct(ref["ml"]["fair_home"]),
        # market gate — both sides
        "edge_ats_home": pp(ats_sides.get("home", {}).get("edge")),
        "edge_ats_away": pp(ats_sides.get("away", {}).get("edge")),
        "edge_ml_home": pp(ml_sides.get("home", {}).get("edge")),
        "edge_ml_away": pp(ml_sides.get("away", {}).get("edge")),
        "verdict_market_ats": ats["verdict"] if ats else None,
        "market_ats_side": side_name(g, ats["side"]) if ats else None,
        "market_ats_line": ats_sides.get(ats["side"], {}).get("line") if ats else None,
        "market_ats_price": ats_sides.get(ats["side"], {}).get("price") if ats else None,
        "verdict_market_ml": ml["verdict"] if ml else None,
        "market_ml_side": side_name(g, ml["side"]) if ml else None,
        "market_ml_price": ml_sides.get(ml["side"], {}).get("price") if ml else None,
        # paper gate — model vs price, never bet
        "verdict_paper_ats": pats["verdict"] if pats else None,
        "paper_ats_side": side_name(g, pats["side"]) if pats else None,
        "paper_ats_line": pats["line"] if pats else None,
        "paper_ats_price": pats["price"] if pats else None,
        "paper_ats_p": pct(pats["p"]) if pats else None,
        "paper_ats_cushion": pp(pats["cushion"]) if pats else None,
        "verdict_paper_ml": pml["verdict"] if pml else None,
        "paper_ml_side": side_name(g, pml["side"]) if pml else None,
        "paper_ml_price": pml["price"] if pml else None,
        "paper_ml_p": pct(pml["p"]) if pml else None,
        "paper_ml_cushion": pp(pml["cushion"]) if pml else None,
        # counts toward the pre-registered look only when BOTH teams are
        # fully off the prior (w_prior = 0 → ≥ 15 regular-season games)
        "paper_counts": bool(g["sh"]["w_prior"] == 0 and g["sa"]["w_prior"] == 0),
        # position (market gate only) — Juan stakes it by entering a stake
        "bet": False,
        "bet_market": pos["market"] if pos else None,
        "bet_side": side_name(g, pos["side"]) if pos else None,
        "bet_line": pos["line"] if pos else None,
        "bet_price": pos["price"] if pos else None,
        "position_note": pos["note"] if pos else "",
        # closes + CLV (filled by auto-grade from the cron snapshot)
        "close_spread": None, "close_spread_price": None,
        "close_spread_price_away": None,
        "close_ml_home": None, "close_ml_away": None,
        "clv_ats_pts": None, "clv_ml_pp": None,
        "clv_paper_ats_pts": None, "clv_paper_ml_pp": None,
        # outcome
        "final_home": None, "final_away": None, "ot": None, "postponed": False,
        "result_ats": "", "result_ml": "",
        "result_paper_ats": "", "result_paper_ml": "", "result_bet": "",
        "pnl": None, "graded_at": None,
        "logged_at": now_et().isoformat(timespec="minutes"),
        "in_play_at_log": g["state"] == "in",
    }
    return row


PRICE_FIELDS = ("fd_source", "fd_spread", "fd_spread_price", "fd_spread_price_away",
                "fd_ml_home", "fd_ml_away", "ref_book", "ref_spread",
                "ref_spread_price", "ref_spread_price_away", "ref_ml_home",
                "ref_ml_away", "fair_margin", "ref_fair_ml_home",
                "edge_ats_home", "edge_ats_away", "edge_ml_home", "edge_ml_away",
                "verdict_market_ats", "market_ats_side", "market_ats_line",
                "market_ats_price", "verdict_market_ml", "market_ml_side",
                "market_ml_price", "verdict_paper_ats", "paper_ats_side",
                "paper_ats_line", "paper_ats_price", "paper_ats_p",
                "paper_ats_cushion", "verdict_paper_ml", "paper_ml_side",
                "paper_ml_price", "paper_ml_p", "paper_ml_cushion",
                "bet_market", "bet_side", "bet_line", "bet_price",
                "position_note", "pick", "prob", "odds", "edge")


def matchup_key(g: dict) -> str:
    """Human-readable label. The NBA has no doubleheaders, but identity is
    NEVER this string — it's espn_event_id (spec §1)."""
    return f"{g['away']} @ {g['home']}"


def grade_row(r: dict, gm: dict, closes: dict, fd_book: str) -> bool:
    """Grade one ungraded row against a completed ESPN game (same event
    ID). Fills finals, OT, results per bucket, the bet's P/L, closes and
    CLV. Returns True if graded. Graded rows are immutable thereafter."""
    if not gm or not gm.get("completed"):
        return False
    fh, fa = int(gm["home_score"]), int(gm["away_score"])
    if fh == fa:
        return False    # not a real final
    r["final_home"], r["final_away"], r["ot"] = fh, fa, bool(gm.get("ot"))
    r["postponed"] = False
    # Closing lines from the cron snapshot (FanDuel, the book you bet at).
    snap = (closes or {}).get(str(r["espn_event_id"])) or {}
    fdq = (snap.get("quotes") or {}).get(fd_book) or {}
    if fdq.get("spreads"):
        sp = fdq["spreads"]
        r["close_spread"] = _float(sp["home"][0])
        r["close_spread_price"] = _int(sp["home"][1]) or None
        r["close_spread_price_away"] = _int(sp["away"][1]) or None
    if fdq.get("h2h"):
        r["close_ml_home"] = _int(fdq["h2h"]["home"]) or None
        r["close_ml_away"] = _int(fdq["h2h"]["away"]) or None

    def ats_result_and_clv(side_nm, line, price):
        sk = side_key(r, side_nm)
        if sk is None or line is None:
            return "", None
        res = grade_ats(sk, line, fh, fa)
        clv = None
        if r.get("close_spread") is not None:
            close_side = (r["close_spread"] if sk == "home" else -r["close_spread"])
            clv = clv_ats_pts(line, close_side)
        return res, clv

    def ml_result_and_clv(side_nm, price):
        sk = side_key(r, side_nm)
        if sk is None:
            return "", None
        res = grade_ml(sk, fh, fa)
        clv = None
        close = r.get("close_ml_home") if sk == "home" else r.get("close_ml_away")
        if is_price(close) and is_price(price):
            clv = clv_ml_pp(price, close)
        return res, clv

    r["result_ats"], r["clv_ats_pts"] = ats_result_and_clv(
        r.get("market_ats_side"), _float(r.get("market_ats_line")),
        r.get("market_ats_price"))
    r["result_ml"], r["clv_ml_pp"] = ml_result_and_clv(
        r.get("market_ml_side"), r.get("market_ml_price"))
    r["result_paper_ats"], r["clv_paper_ats_pts"] = ats_result_and_clv(
        r.get("paper_ats_side"), _float(r.get("paper_ats_line")),
        r.get("paper_ats_price"))
    r["result_paper_ml"], r["clv_paper_ml_pp"] = ml_result_and_clv(
        r.get("paper_ml_side"), r.get("paper_ml_price"))
    # The real bet (stake > 0): P/L in dollars against the budget.
    stake = _float(r.get("stake"), 0.0) or 0.0
    r["bet"] = stake > 0
    res_bet, pnl = "", 0.0
    if r.get("bet_market") and r.get("bet_side"):
        if r["bet_market"] == "ATS":
            res_bet, _ = ats_result_and_clv(r["bet_side"], _float(r.get("bet_line")),
                                            r.get("bet_price"))
        else:
            res_bet, _ = ml_result_and_clv(r["bet_side"], r.get("bet_price"))
        if stake > 0 and res_bet:
            pnl = stake * units_for(res_bet, r.get("bet_price"))
    r["result_bet"] = res_bet
    r["result"] = res_bet            # shared MLB column name
    r["pnl"] = round(pnl, 2) if stake > 0 else None
    r["graded_at"] = now_et().isoformat(timespec="minutes")
    return True


def bucket_stats(rows: list, result_key: str, price_key: str, clv_key: str) -> dict:
    w = sum(1 for r in rows if r.get(result_key) == "W")
    l = sum(1 for r in rows if r.get(result_key) == "L")
    p = sum(1 for r in rows if r.get(result_key) == "Push")
    units = sum(units_for(r.get(result_key), r.get(price_key)) for r in rows)
    clvs = [_float(r.get(clv_key)) for r in rows if _float(r.get(clv_key)) is not None]
    return {"n": w + l + p, "w": w, "l": l, "push": p, "units": round(units, 2),
            "clv": (round(float(np.mean(clvs)), 2) if clvs else None),
            "clv_n": len(clvs),
            "clv_se": (round(float(np.std(clvs, ddof=1) / math.sqrt(len(clvs))), 2)
                       if len(clvs) >= 2 else None)}


FEED_ZERO = {"sp": 0.0, "spj": 0, "spja": 0, "mlh": 0, "mla": 0}


def feed_values(fdq: dict) -> dict:
    """The five form fields as the feed quotes them (zeros where absent)."""
    v = dict(FEED_ZERO)
    if fdq.get("spreads"):
        v["sp"] = float(fdq["spreads"]["home"][0])
        v["spj"] = int(fdq["spreads"]["home"][1])
        v["spja"] = int(fdq["spreads"]["away"][1])
    if fdq.get("h2h"):
        v["mlh"] = int(fdq["h2h"]["home"])
        v["mla"] = int(fdq["h2h"]["away"])
    return v


def clean_value(v):
    """pandas → JSON: NaN / NA / NaT become None; numpy scalars become Python."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.generic):
        return v.item()
    return v


def clean_records(df: pd.DataFrame) -> list:
    return [{k: clean_value(v) for k, v in r.items()} for r in df.to_dict("records")]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 NBA Daily Slate")
    st.caption(f"Season {SEASON - 1}-{SEASON % 100:02d} · ESPN data · "
               f"refreshes every 30 min")
    st.markdown("---")
    st.markdown("##### What decides a bet")
    st.caption(f"**Market gate only**: FanDuel vs {_odds_cfg()['ref'] if _odds_cfg() else 'Pinnacle'} "
               f"(consensus fallback), spread and ML, GOOD at ≥ {MARKET_EDGE_PP}pp. "
               f"The model is context; its picks are **paper** until the "
               f"pre-registered look ({EVAL_LOOK_N} counted picks per bucket "
               f"or {EVAL_LOOK_DATE}, whichever first). No sliders — the "
               f"thresholds live in code, by design.")
    st.markdown("---")
    with st.spinner("Loading season results..."):
        try:
            season_events = fetch_season_events(SEASON, today_et())
            season_err = ""
        except Exception as e:
            season_events, season_err = [], str(e)
    season_log, played_dates, records = build_gamelogs(season_events)
    win_totals, wt_status = load_win_totals()
    with st.spinner("Loading priors..."):
        prior_mov = fetch_prior_mov(PRIOR_SEASON) if len(win_totals) < 30 else {}
    priors = build_priors(win_totals, prior_mov)
    strengths = team_strengths(season_log, priors)

    n_games = sum(len(v) for v in season_log.values()) // 2
    if season_err:
        st.error(f"ESPN season pull failed: {season_err}")
    elif n_games == 0:
        st.info(f"📊 No {SEASON - 1}-{SEASON % 100:02d} regular-season games "
                f"completed yet — running fully on the priors. That is the "
                f"designed opening-night behavior, not an error.")
    else:
        gp = [strengths[t]["n"] for t in NBA_TEAMS]
        st.success(f"✅ {n_games} regular-season games · avg {np.mean(gp):.1f} "
                   f"per team · prior weight avg "
                   f"{np.mean([strengths[t]['w_prior'] for t in NBA_TEAMS]):.0%}")
    n_wt = len(win_totals)
    n_fb = sum(1 for t in NBA_TEAMS if priors[t]["source"] != "win total")
    if wt_status == "missing":
        st.warning(f"⚠️ {WIN_TOTALS_FILE} not found — every prior is the "
                   f"fallback (prior-season MOV × {PRIOR_REGRESS}). Juan enters "
                   f"the 30 FanDuel win totals before Oct 20.")
    elif n_wt < 30:
        st.warning(f"⚠️ Win totals for {n_wt}/30 teams; {n_fb} on the "
                   f"fallback prior (prior-season MOV × {PRIOR_REGRESS}).")
    else:
        st.caption("Priors: 30/30 preseason win totals loaded.")
    if n_fb and not prior_mov and wt_status != "missing":
        st.caption("Fallback prior unavailable (ESPN standings) — those "
                   "teams start at 0.")
    st.caption(f"Model **{MODEL_VERSION}** — frozen; no parameter changes "
               f"before the pre-registered look.")
    if STAKE_UNITS <= 0 or SEASON_BUDGET <= 0:
        st.caption("💵 STAKE_UNITS / SEASON_BUDGET not set in code — positions "
                   "can be logged but not staked.")
    else:
        st.caption(f"💵 Stake ${STAKE_UNITS:,.0f} per position · season budget "
                   f"${SEASON_BUDGET:,.0f}")


# ── Main view ──────────────────────────────────────────────────────────────────
(tab_today,) = st.tabs(["🏀 Today's Games"])

with tab_today:
    today_iso = today_et()
    today_label = today_et("%A, %B %d").replace(" 0", " ")
    st.header(f"Today's Slate — {today_label}")
    st.caption("Every game today run through the frozen model for context, "
               "priced at FanDuel against the sharp reference. The market "
               "gate is the only thing that produces a bet.")

    if st.button("🔄 Refresh data", key=f"{NS}refresh",
                 help="Clear all caches and refetch schedule, results, odds, "
                      "and the pick log"):
        st.cache_data.clear()
        for _k in list(st.session_state.keys()):
            if _k.startswith(("_pick_log", "_closes_")):
                st.session_state.pop(_k, None)
        st.rerun()

    with st.spinner("Fetching today's schedule..."):
        try:
            slate = [g for g in fetch_scoreboard(today_iso) if g["date_et"] == today_iso]
            slate_err = ""
        except Exception as e:
            slate, slate_err = [], str(e)
    if slate_err:
        st.error(f"Could not load the schedule from ESPN: {slate_err}")
        st.stop()
    if not slate:
        st.info("No NBA games scheduled for today. Check back tomorrow!")
        st.stop()

    yesterday_iso = (now_et() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Score each game ────────────────────────────────────────────────────
    slate_results = []
    for g in slate:
        sh, sa = strengths[g["home_id"]], strengths[g["away_id"]]
        # Back-to-back = played on the previous ET date (preseason or
        # regular season both count; spec §3.2). No 3-in-4, no travel.
        home_b2b = (g["home_id"], yesterday_iso) in played_dates
        away_b2b = (g["away_id"], yesterday_iso) in played_dates
        M, hfa, rest = project_margin(sh["strength"], sa["strength"],
                                      g["neutral"], home_b2b, away_b2b)
        ph = p_home_win(M)
        pick_side = "home" if ph >= 0.5 else "away"
        p_pick = ph if pick_side == "home" else 1 - ph
        level = tier_for(p_pick)
        emoji, color = {"High": ("🟢", "#00c07a"), "Moderate": ("🟡", "#f5c842"),
                        "Low": ("🟠", "#f5a623")}[level]
        reasons = []
        wp = max(sh["w_prior"], sa["w_prior"])
        if wp >= 0.99:
            reasons.append("Both ratings are **100% preseason prior** — no "
                           "regular-season games yet.")
        elif wp > 0:
            reasons.append(f"Prior still carries **{wp:.0%}** of a rating here "
                           f"(fades to 0 at {FADE_GAMES} games); this game does "
                           f"NOT count toward the pre-registered look.")
        if home_b2b and away_b2b:
            reasons.append("Both teams on the second night of a back-to-back — "
                           "rest adjustment nets to 0.")
        elif home_b2b or away_b2b:
            tired = g["home"] if home_b2b else g["away"]
            reasons.append(f"**{tired}** is on the second night of a "
                           f"back-to-back (−{B2B_PENALTY} pts).")
        if g["neutral"]:
            reasons.append("Neutral site — no home-court advantage applied.")
        if g["cup_final"]:
            reasons.append("**NBA Cup final** — gradeable as a bet, excluded "
                           "from every team-strength stat.")
        if g["preseason"]:
            reasons.append("**Preseason** — shakedown only. Not logged, not in "
                           "the stats.")
        if not g["regular"] and not g["preseason"]:
            reasons.append(f"**{g['season_type_name'].title()}** — out of the "
                           f"regular-season distribution the model runs on.")
        if p_pick < TIER_MOD_PCT / 100:
            reasons.append(f"The edge is slim ({abs(M):.1f} pts) — inside one "
                           f"possession, where NBA variance (σ≈{SIGMA:g}) dominates.")
        if not reasons:
            reasons.append(f"Opponent-adjusted margin, form, and rest all point "
                           f"to **{side_name(g, pick_side)}**.")
        rh = g["home_record"] or "-".join(map(str, records.get(g["home_id"], [0, 0])))
        ra = g["away_record"] or "-".join(map(str, records.get(g["away_id"], [0, 0])))
        slate_results.append({
            **g, "sh": sh, "sa": sa, "home_b2b": home_b2b, "away_b2b": away_b2b,
            "pred_margin": M, "hfa": hfa, "rest": rest, "p_home": ph,
            "home_pct": round(ph * 100, 1), "away_pct": round((1 - ph) * 100, 1),
            "prob_pick": side_name(g, pick_side), "pick_side": pick_side,
            "model_line_home": -round(M * 2) / 2,
            "conf_level": level, "conf_emoji": emoji, "conf_color": color,
            "conf_reasons": reasons, "rec_h": rh, "rec_a": ra,
        })

    tier_rank = {"High": 0, "Moderate": 1, "Low": 2}
    slate_results.sort(key=lambda x: (x["completed"], tier_rank.get(x["conf_level"], 9),
                                      -abs(x["home_pct"] - 50)))
    n_pre = sum(1 for g in slate_results if g["preseason"])
    st.caption(f"{len(slate_results)} games · sorted by winner confidence then "
               f"edge size" + (f" · {n_pre} preseason (shakedown only)" if n_pre else ""))

    # ── Price gate — all price entry in ONE place, one Enter to apply ─────
    # MLB v4.2 pattern: FanDuel prefills from the feed; the manual override
    # stays for a line the feed missed, and the logged row records which
    # source was used. The reference (Pinnacle / consensus) comes ONLY
    # from the feed — a manual FD line with no feed reference runs the
    # paper gate alone and the market gate reads "no reference".
    upcoming = [g for g in slate_results if not g["completed"] and not g["postponed"]]
    gate = {}          # event_id → {"fd", "ref", "mg", "pg", "pos", "live"}
    feed_by_eid = {}   # event_id → matched feed event (or None)
    live_eids = set()
    ocfg = _odds_cfg()
    credits_left = None
    if upcoming:
        st.subheader("💰 Price gate")
        if ocfg:
            events, credits_left, oerr = fetch_market_odds(
                ocfg["key"], today_et("%Y-%m-%d-%H"))
            if oerr:
                st.error(f"Odds feed: {oerr} — manual FanDuel entry only "
                         f"(no reference → paper gate only).")
            elif events:
                filled = 0
                for g in upcoming:
                    ev, status = match_odds_event(events, g)
                    eid = g["event_id"]
                    if status == "live":
                        live_eids.add(eid)
                        if not st.session_state.get(f"{NS}manual_{eid}"):
                            for f, z in FEED_ZERO.items():
                                st.session_state[f"{NS}{f}_{eid}"] = z
                        continue
                    if not ev:
                        continue
                    feed_by_eid[eid] = ev
                    vals = feed_values(ev["books"].get(ocfg["book"], {}))
                    # Override detection: the user's committed widget values
                    # differ from what the feed last prefilled → a deliberate
                    # manual override; stop the feed overwriting it this
                    # session. (Comparing against the LAST PREFILL, not the
                    # current feed, is what lets a moved line still prefill.)
                    last = st.session_state.get(f"{NS}feedval_{eid}")
                    if last is not None and any(
                            st.session_state.get(f"{NS}{f}_{eid}") != last[f]
                            for f in FEED_ZERO):
                        st.session_state[f"{NS}manual_{eid}"] = True
                    if not st.session_state.get(f"{NS}manual_{eid}"):
                        for f, v in vals.items():
                            st.session_state[f"{NS}{f}_{eid}"] = v
                        st.session_state[f"{NS}feedval_{eid}"] = vals
                    if any(v for v in vals.values()):
                        filled += 1
                st.caption(f"📡 Odds feed: {filled}/{len(upcoming)} games priced from "
                           f"{ocfg['book']}; reference = {ocfg['ref']} (consensus "
                           f"fallback). Pre-game prices only — games underway are "
                           f"left blank. Cached this hour · "
                           f"{credits_left if credits_left is not None else '?'} "
                           f"credits remaining this month.")
            else:
                st.caption("📡 Odds feed returned no NBA events (off day, or "
                           "games already started).")
        else:
            st.warning("📡 No odds feed configured — manual FanDuel entry only "
                       "and NO market gate (it needs the reference from the "
                       "feed). Add an [odds] block (api_key, book, reference) "
                       "to this app's Streamlit Secrets and reboot.")
        # Every field key exists in session state BEFORE its widget is
        # created (feed value or zero), and the widgets below pass no
        # default — so the prefill never trips Streamlit's "default value
        # AND session state" warning, and a game the feed missed starts at 0.
        for g in upcoming:
            for f, z in FEED_ZERO.items():
                st.session_state.setdefault(f"{NS}{f}_{g['event_id']}", z)
        with st.form(f"{NS}odds_form"):
            st.caption("FanDuel's numbers per game: HOME spread (book "
                       "convention, negative = home favored), the juice on "
                       "each side, and both moneylines. Prefilled from the "
                       "feed; edit only what FanDuel shows differently, then "
                       "Apply once. A market counts as priced only when its "
                       "price fields are filled (≥ 100 in magnitude).")
            for g in upcoming:
                eid = g["event_id"]
                fc0, fc1, fc2, fc3, fc4, fc5 = st.columns([2.6, 0.8, 0.8, 0.8, 0.8, 0.8])
                with fc0:
                    st.markdown(
                        f"<div style='padding-top:8px;font-size:13px;'>"
                        f"{matchup_key(g)} <span style='color:#888;'>· "
                        f"{g['tip_label']}</span><br><span style='color:#888;'>"
                        f"model: <b>{g['home']} {fmt_spread(g['model_line_home'])}</b> "
                        f"({g['home_pct']:.1f}% home)"
                        f"{' · PRESEASON' if g['preseason'] else ''}</span></div>",
                        unsafe_allow_html=True)
                with fc1:
                    st.number_input("home spread", min_value=-60.0, max_value=60.0,
                                    step=0.5, format="%.1f",
                                    key=f"{NS}sp_{eid}", help="FanDuel HOME spread")
                with fc2:
                    st.number_input("home juice", min_value=-100000, max_value=100000,
                                    step=5, key=f"{NS}spj_{eid}", help="FanDuel price on the home side of the spread")
                with fc3:
                    st.number_input("away juice", min_value=-100000, max_value=100000,
                                    step=5, key=f"{NS}spja_{eid}", help="FanDuel price on the away side of the spread")
                with fc4:
                    st.number_input("home ML", min_value=-100000, max_value=100000,
                                    step=5, key=f"{NS}mlh_{eid}", help="FanDuel home moneyline")
                with fc5:
                    st.number_input("away ML", min_value=-100000, max_value=100000,
                                    step=5, key=f"{NS}mla_{eid}", help="FanDuel away moneyline")
            st.form_submit_button("Apply FanDuel numbers")
        # Compute every gate from the applied values.
        for g in upcoming:
            eid = g["event_id"]
            if eid in live_eids:
                continue
            sp = _float(st.session_state.get(f"{NS}sp_{eid}"), 0.0)
            spj = _int(st.session_state.get(f"{NS}spj_{eid}"))
            spja = _int(st.session_state.get(f"{NS}spja_{eid}"))
            mlh = _int(st.session_state.get(f"{NS}mlh_{eid}"))
            mla = _int(st.session_state.get(f"{NS}mla_{eid}"))
            has_spread = is_price(spj) or is_price(spja)
            fd = {"spread": sp if has_spread else None,
                  "sp_home": spj if (has_spread and is_price(spj)) else None,
                  "sp_away": spja if (has_spread and is_price(spja)) else None,
                  "ml_home": mlh if is_price(mlh) else None,
                  "ml_away": mla if is_price(mla) else None}
            if not (has_spread or fd["ml_home"] or fd["ml_away"]):
                fd["source"] = "none"
            elif st.session_state.get(f"{NS}manual_{eid}"):
                fd["source"] = "manual"
            elif eid in feed_by_eid:
                fd["source"] = "feed"
            else:
                fd["source"] = "manual"
            ev = feed_by_eid.get(eid)
            ref = reference_lines(ev["books"] if ev else {},
                                  ocfg["book"] if ocfg else "fanduel",
                                  ocfg["ref"] if ocfg else "pinnacle")
            mg = market_gate(fd, ref)
            pg = paper_gate(g["pred_margin"], fd)
            pos = position_from_gate(mg)
            gate[eid] = {"fd": fd, "ref": ref, "mg": mg, "pg": pg, "pos": pos}

        # Auto-sync: the panel is the source of truth for TODAY's prices.
        # Rows already logged today (this version, ungraded, UNSTAKED) take
        # the panel's prices/verdicts — no second data entry in the
        # tracker. A staked row is frozen: its logged basis is the bet.
        if gate:
            _log = load_pick_log()
            _by_eid = {g["event_id"]: g for g in upcoming}
            _synced = 0
            for _r in _log:
                _eid = str(_r.get("espn_event_id", ""))
                if (_eid not in gate or _r.get("version") != MODEL_VERSION
                        or _r.get("graded_at") or (_float(_r.get("stake"), 0) or 0) > 0):
                    continue
                _gt = gate[_eid]
                _fresh = build_row(_by_eid[_eid], _gt["fd"], _gt["ref"], _gt["mg"],
                                   _gt["pg"], _gt["pos"])
                if any(_r.get(k) != _fresh.get(k) for k in PRICE_FIELDS):
                    for k in PRICE_FIELDS:
                        _r[k] = _fresh[k]
                    _synced += 1
            if _synced:
                save_pick_log(_log)
                st.caption(f"🔄 Synced {_synced} row(s) in today's tracker from "
                           f"the price panel.")

        # Decision board — one line per priced game, best market edge first.
        board = []
        for g in upcoming:
            gt = gate.get(g["event_id"])
            if not gt or gt["fd"]["source"] == "none":
                continue
            mg, pg, pos = gt["mg"], gt["pg"], gt["pos"]
            def _mk(m):
                if not m:
                    return "—", None
                sd = m["sides"][m["side"]]
                ln = f" {fmt_spread(sd['line'])}" if sd["line"] is not None else ""
                return (f"{VERDICT_ICON[m['verdict']]} {side_name(g, m['side'])}{ln} "
                        f"{fmt_price(sd['price'])} ({sd['edge']*100:+.1f}pp)"), sd["edge"]
            ats_txt, ats_e = _mk(mg["ats"])
            ml_txt, ml_e = _mk(mg["ml"])
            def _pp(p, spread=False):
                if not p or p["verdict"] is None:
                    return "—"
                ln = f" {fmt_spread(p['line'])}" if spread else ""
                return (f"{VERDICT_ICON[p['verdict']]} {side_name(g, p['side'])}{ln} "
                        f"{fmt_price(p['price'])} ({p['cushion']*100:+.1f}pp)")
            board.append({
                "Matchup": matchup_key(g), "Tip": g["tip_label"],
                "Market ATS": ats_txt, "Market ML": ml_txt,
                "Position": (f"{pos['market']}: {side_name(g, pos['side'])}"
                             + (f" {fmt_spread(pos['line'])}" if pos["line"] is not None else "")
                             + f" {fmt_price(pos['price'])}") if pos else "—",
                "Ref": gt["ref"]["ats"]["src"] or gt["ref"]["ml"]["src"] or "none",
                "FD src": gt["fd"]["source"],
                "Paper ATS": _pp(pg["ats"], True), "Paper ML": _pp(pg["ml"]),
                "Model": f"{g['conf_level']} · {g['prob_pick']} {max(g['home_pct'], g['away_pct']):.0f}%",
                "_e": max([e for e in (ats_e, ml_e) if e is not None] or [-9]),
            })
        if board:
            st.markdown(
                "<div style='font-size:12px;color:#aaa;margin-bottom:6px;'>"
                "<b>How the gate decides:</b> a good bet is a price better than "
                "the true chance. The one estimate of that chance the suite's "
                "data supports is the sharp market's — so the market gate asks "
                "whether FanDuel's number beats the reference (Pinnacle, else "
                f"consensus) by ≥ {MARKET_EDGE_PP}pp, on both sides of both "
                "markets. Spread + ML GOOD on the same side = one position "
                "(spread). The model's own picks are shown as <i>paper</i> and "
                "logged; closing-line value in the tracker is their scoreboard."
                "</div>", unsafe_allow_html=True)
            board.sort(key=lambda r: -r["_e"])
            st.dataframe(pd.DataFrame(board).drop(columns=["_e"]),
                         hide_index=True, width="stretch")
            n_pos = sum(1 for r in board if r["Position"] != "—")
            if n_pos == 0:
                st.markdown("<div style='font-size:13px;color:#f5c842;"
                            "font-weight:700;'>No plays today — sitting out IS "
                            "the play. FanDuel offered nothing better than the "
                            "reference.</div>", unsafe_allow_html=True)
            else:
                st.caption(f"{n_pos} position(s) of {len(board)} priced games.")
        elif ocfg:
            st.caption("No FanDuel prices yet for today's games.")

    # ── Render each game ───────────────────────────────────────────────────
    for game in slate_results:
        color = game["conf_color"]
        eid = game["event_id"]
        if game["completed"]:
            hs, as_ = game["home_score"], game["away_score"]
            winner = game["home"] if hs > as_ else game["away"]
            status_badge = (
                f"<span style='background:rgba(0,192,122,0.15);color:#00c07a;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>Final"
                f"{' (OT)' if game['ot'] else ''} · {winner} won "
                f"{max(hs, as_)}-{min(hs, as_)}</span>")
        elif game["postponed"]:
            status_badge = ("<span style='background:rgba(255,165,0,0.15);color:#f5a623;"
                            "font-size:11px;padding:2px 8px;border-radius:4px;'>"
                            "Postponed</span>")
        elif game["state"] == "in":
            status_badge = (
                f"<span style='background:rgba(255,82,82,0.15);color:#ff5252;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>🔴 Live · "
                f"{game['status_detail']}</span>")
        else:
            status_badge = (
                f"<span style='background:rgba(61,139,255,0.15);color:#3d8bff;"
                f"font-size:11px;padding:2px 8px;border-radius:4px;'>"
                f"{game['tip_label']}</span>")
        tags = []
        if game["preseason"]:
            tags.append("PRESEASON · shakedown only — not logged, not in stats")
        elif not game["regular"]:
            tags.append(f"{game['season_type_name'].upper()} · out-of-distribution")
        if game["cup_final"]:
            tags.append("NBA CUP FINAL · excluded from team strength")
        if game["neutral"]:
            tags.append(f"🏟 Neutral site — {game['venue'] or 'no HFA'}")
        if game["home_b2b"] and game["away_b2b"]:
            tags.append("😴 Both on a back-to-back (nets to 0)")
        elif game["home_b2b"]:
            tags.append(f"😴 {game['home_abbr']} on a back-to-back (−{B2B_PENALTY})")
        elif game["away_b2b"]:
            tags.append(f"😴 {game['away_abbr']} on a back-to-back (−{B2B_PENALTY})")
        tag_html = "".join(
            f"<span style='font-size:11px;color:#aaa;margin-left:6px;'>{t}</span>"
            for t in tags)
        sh, sa = game["sh"], game["sa"]
        reasons_html = "".join(
            f"<div style='margin:2px 0;font-size:12px;color:#aaa;'>&bull; {r}</div>"
            for r in game["conf_reasons"])
        home_prob_color = "#00c07a" if game["pick_side"] == "home" else "#aaa"
        home_prob_weight = "800" if game["pick_side"] == "home" else "400"
        away_prob_color = "#00c07a" if game["pick_side"] == "away" else "#aaa"
        away_prob_weight = "800" if game["pick_side"] == "away" else "400"

        def _src(s):
            if s["w_prior"] >= 0.99:
                return f"100% prior ({s['prior_src']})"
            if s["w_prior"] > 0:
                return f"{s['w_prior']:.0%} prior / {1 - s['w_prior']:.0%} season"
            return f"season (form {s['w_form']:.0%})"

        card_html = (
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;'
            f'border-left:4px solid {color};border-radius:12px;padding:18px 22px;margin-bottom:16px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'flex-wrap:wrap;gap:8px;margin-bottom:14px;">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
            f'<span style="font-size:17px;font-weight:700;">'
            f'{game["away"]} <span style="color:#555;font-size:13px;font-weight:400;">({game["rec_a"]})</span>'
            f' <span style="color:#555;margin:0 6px;">@</span> '
            f'{game["home"]} <span style="color:#555;font-size:13px;font-weight:400;">({game["rec_h"]})</span>'
            f'</span> {status_badge}{tag_html}</div>'
            f'<span style="font-size:13px;font-weight:700;color:{color};">'
            f'{game["conf_emoji"]} {game["conf_level"]} confidence</span></div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1.4fr 1.4fr;gap:16px;margin-bottom:14px;">'
            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Win probability</div>'
            f'<div style="font-size:14px;">'
            f'<span style="color:{home_prob_color};font-weight:{home_prob_weight};">'
            f'{game["home"]} {game["home_pct"]}%</span><br>'
            f'<span style="color:{away_prob_color};font-weight:{away_prob_weight};">'
            f'{game["away"]} {game["away_pct"]}%</span></div></div>'
            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">Model line</div>'
            f'<div style="font-size:14px;font-weight:700;color:{color};">'
            f'{game["home"]} {fmt_spread(game["model_line_home"])}'
            f'<br><span style="font-size:11px;font-weight:400;color:#888;">'
            f'Proj margin {game["pred_margin"]:+.1f} · HFA {game["hfa"]:+g} · '
            f'rest {game["rest"]:+g}</span></div></div>'
            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">{game["home_abbr"]} strength (pts)</div>'
            f'<div style="font-size:11px;color:#aaa;">'
            f'Rating: <span style="color:#ccc;font-weight:700;">{sh["strength"]:+.1f}</span> '
            f'<span style="color:#666;">({_src(sh)})</span><br>'
            f'Prior {sh["prior"]:+.1f} <span style="color:#666;">{sh["prior_detail"]}</span><br>'
            f'Season adj {sh["season_adj"]:+.1f} (raw {sh["raw_mpg"]:+.1f}, n={sh["n"]}) · '
            f'form {sh["form_adj"]:+.1f}</div></div>'
            f'<div><div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:#666;margin-bottom:4px;">{game["away_abbr"]} strength (pts)</div>'
            f'<div style="font-size:11px;color:#aaa;">'
            f'Rating: <span style="color:#ccc;font-weight:700;">{sa["strength"]:+.1f}</span> '
            f'<span style="color:#666;">({_src(sa)})</span><br>'
            f'Prior {sa["prior"]:+.1f} <span style="color:#666;">{sa["prior_detail"]}</span><br>'
            f'Season adj {sa["season_adj"]:+.1f} (raw {sa["raw_mpg"]:+.1f}, n={sa["n"]}) · '
            f'form {sa["form_adj"]:+.1f}</div></div></div>'
            f'<div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:10px;">'
            f'{reasons_html}</div></div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Gate verdicts under the card (read-only; entry is in the panel)
        if game["completed"] or game["postponed"]:
            continue
        gt = gate.get(eid)
        if eid in live_eids or game["state"] == "in":
            st.markdown("<div style='font-size:11px;color:#c77;margin:-6px 0 14px 2px;'>"
                        "🔴 Game underway — no pre-game price; not gated</div>",
                        unsafe_allow_html=True)
            continue
        if not gt or gt["fd"]["source"] == "none":
            st.markdown("<div style='font-size:11px;color:#555;margin:-6px 0 14px 2px;'>"
                        "No FanDuel price — add it in the price panel above</div>",
                        unsafe_allow_html=True)
            continue
        lines = []
        for label, m in (("ATS", gt["mg"]["ats"]), ("ML", gt["mg"]["ml"])):
            if not m:
                lines.append(f"<span style='color:#888;'>{label}&nbsp; no reference "
                             f"in the feed — market gate can't run</span>")
                continue
            sd = m["sides"][m["side"]]
            ln = f" {fmt_spread(sd['line'])}" if sd["line"] is not None else ""
            other = m["sides"].get("away" if m["side"] == "home" else "home")
            oth = (f" · other side {other['edge']*100:+.1f}pp" if other else "")
            lines.append(
                f"<span style='color:#888;'>{label}&nbsp;</span>"
                f"<span style='color:{VERDICT_COLOR[m['verdict']]};font-weight:700;'>"
                f"{VERDICT_ICON[m['verdict']]} {m['verdict']}</span>"
                f"<span style='color:#888;'> {side_name(game, m['side'])}{ln} at "
                f"{fmt_price(sd['price'])} — {m['src']} fair {sd['fair']*100:.1f}% vs "
                f"needs {sd['be']*100:.1f}% (edge {sd['edge']*100:+.1f}pp){oth}</span>")
        pos = gt["pos"]
        if pos:
            ln = f" {fmt_spread(pos['line'])}" if pos["line"] is not None else ""
            lines.append(f"<span style='color:#00c07a;font-weight:700;'>POSITION: "
                         f"{pos['market']} {side_name(game, pos['side'])}{ln} "
                         f"{fmt_price(pos['price'])}</span>"
                         + (f"<span style='color:#888;'> — {pos['note']}</span>"
                            if pos["note"] else ""))
        for label, p, spread in (("Paper ATS", gt["pg"]["ats"], True),
                                 ("Paper ML", gt["pg"]["ml"], False)):
            if not p or p["verdict"] is None:
                continue
            ln = f" {fmt_spread(p['line'])}" if spread else ""
            lines.append(
                f"<span style='color:#888;'>{label}&nbsp;</span>"
                f"<span style='color:{VERDICT_COLOR[p['verdict']]};'>"
                f"{VERDICT_ICON[p['verdict']]} {p['verdict']}</span>"
                f"<span style='color:#888;'> {side_name(game, p['side'])}{ln} at "
                f"{fmt_price(p['price'])} — model {p['p']*100:.1f}% vs needs "
                f"{p['be']*100:.1f}% (cushion {p['cushion']*100:+.1f}pp) · "
                f"<i>paper, never bet</i></span>")
        st.markdown("<div style='font-size:12px;margin:-6px 0 14px 2px;'>"
                    + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PICK TRACKER — forward measurement, every game, bet or not.
    # Same instrument as the siblings: logs the live gate exactly as
    # displayed (no look-ahead possible), tagged with MODEL_VERSION. Four
    # buckets: market_ats, market_ml (what produces bets), paper_ats,
    # paper_ml (the model's own picks — measured, never bet in v1.0).
    # Un-bet games are the control group. Graded rows are immutable.
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📌 Pick Tracker")
    show_flash()
    _cfg = _gh_cfg()
    if _cfg:
        st.caption(f"🗄 Log storage: GitHub — {_cfg['repo']} @ {_cfg['branch']} "
                   f"/ {PICK_LOG_FILE} (survives Streamlit Cloud reboots)")
    else:
        st.caption("🗄 Log storage: local file — fine on your machine, but "
                   "WIPED on Streamlit Cloud reboots. Add [github] secrets "
                   "to persist (see comment above _gh_cfg in the code).")

    log = load_pick_log()
    fd_book = ocfg["book"] if ocfg else "fanduel"
    c_log1, c_log2, c_log3, c_log4 = st.columns([1, 1, 1, 1.2])
    with c_log1:
        if st.button("Log today's slate to tracker", key=f"{NS}log"):
            slate_eids = {g["event_id"] for g in slate_results if not g["preseason"]}
            # Supersede: UNGRADED rows from an older model version for games
            # on today's slate are replaced by the current model's read.
            # GRADED rows are immutable forever, whatever version wrote them.
            before = len(log)
            log[:] = [r for r in log if not (
                r.get("version") != MODEL_VERSION and not r.get("graded_at")
                and str(r.get("espn_event_id")) in slate_eids)]
            replaced = before - len(log)
            existing = {str(r.get("espn_event_id")) for r in log
                        if r.get("version") == MODEL_VERSION}
            added = skipped_pre = skipped_final = 0
            for g in slate_results:
                eid = g["event_id"]
                if g["preseason"]:
                    skipped_pre += 1
                    continue
                if g["completed"]:
                    # A pick logged after the result exists is look-ahead.
                    skipped_final += 1
                    continue
                if eid in existing:
                    continue
                gt = gate.get(eid)
                if gt:
                    row = build_row(g, gt["fd"], gt["ref"], gt["mg"], gt["pg"], gt["pos"])
                else:
                    fd0 = {"source": "none", "spread": None, "sp_home": None,
                           "sp_away": None, "ml_home": None, "ml_away": None}
                    row = build_row(g, fd0, reference_lines({}, fd_book, "pinnacle"),
                                    {"ats": None, "ml": None},
                                    paper_gate(g["pred_margin"], fd0), None)
                row["postponed"] = bool(g["postponed"])
                log.append(row)
                added += 1
            save_pick_log(log)
            msg = (f"Logged {added} new game(s)." if added
                   else "Today's slate is already logged — prices applied "
                        "above sync into it automatically.")
            if replaced:
                msg += (f" Replaced {replaced} ungraded row(s) from a superseded "
                        f"model version.")
            if skipped_pre:
                msg += f" Skipped {skipped_pre} preseason game(s) (shakedown only)."
            if skipped_final:
                msg += (f" Skipped {skipped_final} already-final game(s) — "
                        f"look-ahead; only games still biddable enter the record.")
            flash(msg)
            st.rerun()
    with c_log2:
        # Auto-grade pulls finals from ESPN by EVENT ID, fills the FanDuel
        # close from the cron snapshot on the data branch, and computes CLV:
        # ATS in spread points, ML in probability points (NFL pattern).
        # Postponed → row stays ungraded with a flag; re-grades when the
        # same event ID completes (rescheduled games are found in the
        # season index, whatever date they moved to).
        if st.button("⚡ Auto-grade finished games", key=f"{NS}grade"):
            graded = flagged = 0
            season_by_id = {g["event_id"]: g for g in season_events}
            for r in log:
                if r.get("graded_at") or str(r.get("date", "")) > today_iso:
                    continue
                eid = str(r.get("espn_event_id", ""))
                if not eid:
                    continue
                gm = None
                try:
                    gm = next((x for x in fetch_scoreboard(r["date"])
                               if x["event_id"] == eid), None)
                except Exception:
                    gm = None
                if gm is None or not gm["completed"]:
                    alt = season_by_id.get(eid)
                    if alt is not None and alt["completed"]:
                        gm = alt
                if gm is None:
                    continue
                if gm["postponed"] and not gm["completed"]:
                    if not r.get("postponed"):
                        r["postponed"] = True
                        flagged += 1
                    continue
                closes = load_closes(r["date"])
                if str(eid) not in closes and gm["date_et"] != r["date"]:
                    closes = load_closes(gm["date_et"])
                if grade_row(r, gm, closes, fd_book):
                    if gm["date_et"] != r["date"]:
                        r["rescheduled_to"] = gm["date_et"]
                    graded += 1
            if graded or flagged:
                save_pick_log(log)
                flash(f"Auto-graded {graded} game(s)."
                      + (f" Flagged {flagged} postponed (stay ungraded until "
                         f"rescheduled)." if flagged else ""))
                st.rerun()
            else:
                st.info("Nothing new to grade yet.")
    with c_log3:
        # Stakes today's market-gate positions at STAKE_UNITS — the
        # mechanical execution of the gate, one position per game, never
        # beyond SEASON_BUDGET. Both values live in code (spec §6).
        if st.button("💵 Stake today's positions", key=f"{NS}stake"):
            if STAKE_UNITS <= 0 or SEASON_BUDGET <= 0:
                flash("STAKE_UNITS / SEASON_BUDGET are 0 — set them in code "
                      "before the first bet. Nothing staked.", "error")
                st.rerun()
            spent = sum((_float(r.get("stake"), 0) or 0) for r in log
                        if r.get("version") == MODEL_VERSION)
            staked = refused = 0
            for r in log:
                if (r.get("version") != MODEL_VERSION or r.get("date") != today_iso
                        or r.get("graded_at") or not r.get("bet_market")
                        or (_float(r.get("stake"), 0) or 0) > 0):
                    continue
                if spent + STAKE_UNITS > SEASON_BUDGET:
                    refused += 1
                    continue
                r["stake"] = float(STAKE_UNITS)
                r["bet"] = True
                r["odds"] = r.get("bet_price") or 0
                spent += STAKE_UNITS
                staked += 1
            if staked:
                save_pick_log(log)
            flash(f"Staked {staked} position(s) at ${STAKE_UNITS:,.0f}."
                  + (f" Refused {refused} — season budget exhausted. No reload: "
                     f"that was the deal before opening night." if refused else ""),
                  "success" if staked else "warning")
            st.rerun()
    with c_log4:
        st.caption("Daily loop: open the app (prices prefill), log the slate, "
                   "stake the positions you actually bet, auto-grade the next "
                   "morning. Every game is graded — bet or not. Paper buckets "
                   "are measured by CLV; the look is pre-registered.")

    EDITABLE = ["stake", "bet_market", "bet_side", "bet_line", "bet_price"]
    COLUMN_ORDER = ["date", "matchup", "tip_et", "tier", "model_margin", "p_home_win",
                    "fd_source", "fd_spread", "fd_spread_price", "fd_ml_home", "fd_ml_away",
                    "ref_book", "ref_spread", "fair_margin",
                    "verdict_market_ats", "market_ats_side", "market_ats_line",
                    "edge_ats_home", "edge_ats_away",
                    "verdict_market_ml", "market_ml_side", "edge_ml_home", "edge_ml_away",
                    "bet_market", "bet_side", "bet_line", "bet_price", "stake",
                    "position_note", "result_bet", "pnl",
                    "result_ats", "result_ml", "clv_ats_pts", "clv_ml_pp",
                    "verdict_paper_ats", "paper_ats_side", "paper_ats_line",
                    "paper_ats_cushion", "result_paper_ats", "clv_paper_ats_pts",
                    "verdict_paper_ml", "paper_ml_side", "paper_ml_cushion",
                    "result_paper_ml", "clv_paper_ml_pp", "paper_counts",
                    "close_spread", "close_spread_price", "close_ml_home", "close_ml_away",
                    "final_home", "final_away", "ot", "postponed", "home_b2b", "away_b2b",
                    "neutral_site", "cup_final", "season_type", "version", "graded_at",
                    "espn_event_id"]
    if log:
        df = pd.DataFrame(log)
        for c in COLUMN_ORDER:
            if c not in df.columns:
                df[c] = None
        df = df.sort_values(["date", "tip_et"], ascending=[False, True],
                            kind="stable").reset_index(drop=True)
        edited = st.data_editor(
            df,
            column_order=[c for c in COLUMN_ORDER if c in df.columns],
            disabled=[c for c in df.columns if c not in EDITABLE],
            column_config={
                "stake": st.column_config.NumberColumn(
                    "stake $", help="Dollars actually wagered on the position. "
                    "0 = not bet (control row). Drives P/L against SEASON_BUDGET.",
                    step=5),
                "bet_market": st.column_config.SelectboxColumn(
                    "bet market", options=["", "ATS", "ML"],
                    help="Prefilled from the market gate's position. Change only "
                         "if you took the other market."),
                "bet_side": st.column_config.TextColumn(
                    "bet side", help="Team name of the side you bet"),
                "bet_line": st.column_config.NumberColumn(
                    "bet line", help="Spread you got (side view), ATS only", step=0.5),
                "bet_price": st.column_config.NumberColumn(
                    "bet price", help="American price you got", step=5),
                "verdict_market_ats": st.column_config.TextColumn("mkt ATS"),
                "verdict_market_ml": st.column_config.TextColumn("mkt ML"),
                "verdict_paper_ats": st.column_config.TextColumn("paper ATS"),
                "verdict_paper_ml": st.column_config.TextColumn("paper ML"),
                "clv_ats_pts": st.column_config.NumberColumn(
                    "CLV ATS (pts)", help="Market-gate side: logged line − FanDuel "
                    "close, side view. Positive = beat the close."),
                "clv_ml_pp": st.column_config.NumberColumn(
                    "CLV ML (pp)", help="Market-gate side: BE(close) − BE(logged). "
                    "Positive = beat the close."),
                "paper_counts": st.column_config.CheckboxColumn(
                    "counts", help="Both teams fully off the prior (≥ 15 games) — "
                    "counts toward the pre-registered look"),
            },
            num_rows="fixed", hide_index=True, width="stretch", height=320,
            key=f"{NS}editor")
        if st.button("Save grades", key=f"{NS}save"):
            rows_out = clean_records(edited)
            for r in rows_out:
                stake = _float(r.get("stake"), 0.0) or 0.0
                r["stake"] = stake
                r["bet"] = stake > 0
                if r.get("bet_side"):
                    tid = team_id_for(r["bet_side"])
                    if tid in (str(r.get("home_id")), str(r.get("away_id"))):
                        r["bet_side"] = team_name(tid)
                if r.get("bet_market") == "":
                    r["bet_market"] = None
                if r.get("bet_price"):
                    r["odds"] = r["bet_price"]
            save_pick_log(rows_out)
            flash("Saved.")
            st.rerun()

        # ── Summary — current model version only ──────────────────────────
        all_rows = clean_records(edited)
        cur = [r for r in all_rows
               if r.get("version") == MODEL_VERSION and r.get("graded_at")]
        n_all = sum(1 for r in all_rows if r.get("version") == MODEL_VERSION)
        st.markdown(f"**{MODEL_VERSION}** — {n_all} game(s) logged, "
                    f"{len(cur)} graded.")
        if cur:
            def _rec(b):
                return f"{b['w']}–{b['l']}" + (f"–{b['push']}" if b["push"] else "")
            def _clv(b, unit):
                if b["clv"] is None:
                    return "—"
                se = f" ±{b['clv_se']}" if b["clv_se"] is not None else ""
                return f"{b['clv']:+.2f}{se} {unit} (n={b['clv_n']})"
            rows = []
            # Market gate — what produces bets. GOOD rows vs the rest.
            for label, vk, rk, pk, ck, unit in (
                    ("market_ats", "verdict_market_ats", "result_ats",
                     "market_ats_price", "clv_ats_pts", "pts"),
                    ("market_ml", "verdict_market_ml", "result_ml",
                     "market_ml_price", "clv_ml_pp", "pp")):
                good = [r for r in cur if r.get(vk) == VERDICT_GOOD and r.get(rk)]
                rest = [r for r in cur if r.get(vk) in (VERDICT_THIN, VERDICT_AWAY)
                        and r.get(rk)]
                for sub, rs in (("GOOD PRICE", good), ("control (thin / stay away)", rest)):
                    if not rs:
                        continue
                    b = bucket_stats(rs, rk, pk, ck)
                    rows.append({"bucket": f"{label} · {sub}", "n": b["n"],
                                 "record": _rec(b), "units (flat 1u)": b["units"],
                                 "avg CLV": _clv(b, unit), "toward look": "—"})
            # Paper gate — the model's picks. "all" and "counts" (post-fade).
            for label, vk, rk, pk, ck, unit in (
                    ("paper_ats", "verdict_paper_ats", "result_paper_ats",
                     "paper_ats_price", "clv_paper_ats_pts", "pts"),
                    ("paper_ml", "verdict_paper_ml", "result_paper_ml",
                     "paper_ml_price", "clv_paper_ml_pp", "pp")):
                good = [r for r in cur if r.get(vk) == VERDICT_GOOD and r.get(rk)]
                counts = [r for r in good if r.get("paper_counts")]
                for sub, rs in (("GOOD · all", good), ("GOOD · counts", counts)):
                    if not rs and sub.endswith("all"):
                        continue
                    b = bucket_stats(rs, rk, pk, ck)
                    rows.append({"bucket": f"{label} · {sub}", "n": b["n"],
                                 "record": _rec(b), "units (flat 1u)": b["units"],
                                 "avg CLV": _clv(b, unit),
                                 "toward look": (f"{b['n']}/{EVAL_LOOK_N}"
                                                 if sub.endswith("counts") else "—")})
            allml = [r for r in cur if r.get("result_paper_ml")]
            if allml:
                b = bucket_stats(allml, "result_paper_ml", "paper_ml_price",
                                 "clv_paper_ml_pp")
                rows.append({"bucket": "model pick · every game (context)",
                             "n": b["n"], "record": _rec(b),
                             "units (flat 1u)": "—", "avg CLV": "—",
                             "toward look": "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Units: flat 1u at the logged FanDuel price on the bucket's "
                       "side; pushes returned. CLV: ATS in spread points, ML in "
                       "probability points, positive = beat the FanDuel close. "
                       "Win rate over 200 picks has SE ≈ 3.5pp — it cannot acquit "
                       "or convict the model. CLV is the verdict; the record is a "
                       "sanity check.")
            # Positions and the season budget (real money).
            bets = [r for r in all_rows
                    if r.get("version") == MODEL_VERSION
                    and (_float(r.get("stake"), 0) or 0) > 0]
            if bets or SEASON_BUDGET > 0:
                spent = sum(_float(r.get("stake"), 0) or 0 for r in bets)
                pl = sum(_float(r.get("pnl"), 0) or 0 for r in bets if r.get("graded_at"))
                bw = sum(1 for r in bets if r.get("result_bet") == "W")
                bl = sum(1 for r in bets if r.get("result_bet") == "L")
                bp = sum(1 for r in bets if r.get("result_bet") == "Push")
                if SEASON_BUDGET > 0:
                    left = SEASON_BUDGET - spent
                    col = "#00c07a" if left > 0 else "#ff5252"
                    st.markdown(f"**Positions** — {len(bets)} staked, {bw}–{bl}"
                                f"{f'–{bp}' if bp else ''} · P/L ${pl:+,.0f} · "
                                f"staked ${spent:,.0f} of ${SEASON_BUDGET:,.0f} "
                                f"(<span style='color:{col};'>${left:,.0f} "
                                f"remaining</span>)", unsafe_allow_html=True)
                    if left <= 0:
                        st.error("Budget spent. No reload — that was the deal "
                                 "before opening night.")
                else:
                    st.warning(f"Real stakes logged (${spent:,.0f}) but "
                               f"SEASON_BUDGET is 0. Set it in code before the "
                               f"next bet.")
        # Pre-registered look — progress, never a peek.
        days_left = (datetime.date.fromisoformat(EVAL_LOOK_DATE) - now_et().date()).days
        st.caption(f"📅 Pre-registered look (nba_evaluation_plan.md): the earlier "
                   f"of {EVAL_LOOK_N} counted GOOD paper picks in a bucket or "
                   f"{EVAL_LOOK_DATE} ({days_left} days). Look ONCE. Pass = mean "
                   f"CLV ≥ +0.5 pts (paper_ats) / ≥ +1.0pp (paper_ml) with the 95% "
                   f"interval excluding zero, record not contradicting. Pass → "
                   f"that bucket's model gate is ADDED beside the market gate at "
                   f"0.5× stake as nba-v1.1. Fail → stays paper. Never the model "
                   f"alone.")
    else:
        st.caption("No games logged yet. Log today's slate to start the record.")

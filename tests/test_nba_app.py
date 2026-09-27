"""
AppTest harness for nba_app.py — spec §8.3.

Runs the real app under streamlit.testing.v1.AppTest against a deterministic
mock ESPN + mock Odds feed (requests.get is patched; the clock is pinned via
NBA_APP_CLOCK). Covers: overtime; back-to-back home / away / both; neutral
site; postponed (and re-grade under the same event ID); preseason game on
the slate (excluded from stats and log); a 1 pm weekend tip; a day with no
games; in-play prices in the feed (filtered); Pinnacle missing (consensus
fallback); whole-number spread push; NBA Cup final flag (slate + history
exclusion); both-markets-GOOD same side (position rule). Plus pure tests of
the model/gate math and the closes cron's spending decision.

Run:  python -m pytest tests/test_nba_app.py -q
"""
import datetime
import json
import math
import os
import random
import shutil
import sys
from zoneinfo import ZoneInfo

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPTS = os.path.join(REPO, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from nba_teams import NBA_TEAMS, team_name, team_abbr  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = datetime.timezone.utc
TEAM_IDS = [str(i) for i in range(1, 31)]

SAT = "2026-11-21"      # a Saturday
SUN = "2026-11-22"
CLOCK_SAT = "2026-11-21T10:00:00-05:00"
CLOCK_SUN = "2026-11-22T09:00:00-05:00"
CLOCK_WED = "2026-11-25T09:00:00-05:00"   # after the postponed game replays
CLOCK_OFF = "2026-11-23T09:00:00-05:00"   # no games


# ── Mock ESPN season ───────────────────────────────────────────────────────────
def _iso(dt: datetime.datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def espn_event(eid, home, away, tip_utc, status="STATUS_SCHEDULED", state="pre",
               completed=False, period=0, season_type=2, neutral=False, notes=None,
               hs=0, as_=0, detail="", venue="Mock Arena"):
    comps = []
    for ha, tid, sc in (("home", home, hs), ("away", away, as_)):
        comps.append({"homeAway": ha, "score": str(sc),
                      "team": {"id": tid, "displayName": team_name(tid),
                               "abbreviation": team_abbr(tid)},
                      "records": [{"type": "total", "summary": "10-5"}]})
    return {"id": str(eid), "date": tip_utc, "name": f"{team_name(away)} at {team_name(home)}",
            "season": {"year": 2027, "type": season_type},
            "status": {"period": period,
                       "type": {"name": status, "state": state, "completed": completed,
                                "shortDetail": detail}},
            "competitions": [{"id": str(eid), "neutralSite": neutral,
                              "notes": ([{"type": "event", "headline": notes}] if notes else []),
                              "venue": {"fullName": venue}, "competitors": comps}]}


def build_history():
    """Regular season Oct 20 → Nov 20 (yesterday). Deterministic. Teams 29
    and 30 play rarely (→ still on the prior). Yesterday's games are fixed
    so the back-to-back cases are exact. Includes one OT game, a preseason
    week, and a completed NBA Cup final that must NOT enter the stats."""
    rng = random.Random(7)
    strength = {t: rng.gauss(0, 6) for t in TEAM_IDS}
    events, eid = [], 401800000

    def final(home, away, tip, season_type=2, neutral=False, notes=None, ot=False,
              score=None):
        nonlocal eid
        eid += 1
        if score is None:
            m = strength[home] - strength[away] + 2.5 + rng.gauss(0, 12)
            hs, as_ = round((224 + m) / 2), round((224 - m) / 2)
            if hs == as_:
                hs += 1
        else:
            hs, as_ = score
        return espn_event(eid, home, away, tip, "STATUS_FINAL", "post", True,
                          5 if ot else 4, season_type, neutral, notes, hs, as_, "Final")

    # Preseason (type 1): Oct 5-7, four games a night.
    for d in range(5, 8):
        teams = TEAM_IDS[:]
        rng.shuffle(teams)
        for i in range(0, 8, 2):
            tip = datetime.datetime(2026, 10, d, 19, 0, tzinfo=ET)
            events.append(final(teams[i], teams[i + 1], _iso(tip), season_type=1))
    # Regular season: Oct 20 → Nov 19, eight games a night.
    day = datetime.date(2026, 10, 20)
    ot_done = False
    while day <= datetime.date(2026, 11, 19):
        teams = [t for t in TEAM_IDS if t not in ("29", "30") or rng.random() < 0.4]
        rng.shuffle(teams)
        for i in range(0, 16, 2):
            if i + 1 >= len(teams):
                break
            tip = datetime.datetime(day.year, day.month, day.day, 19, 0, tzinfo=ET)
            ot = (not ot_done) and day == datetime.date(2026, 11, 3)
            events.append(final(teams[i], teams[i + 1], _iso(tip), ot=ot))
            ot_done = ot_done or ot
        day += datetime.timedelta(days=1)
    # Completed NBA Cup final (Nov 15, neutral, absurd score): flagged by
    # its notes headline; must be excluded from team strength.
    events.append(final("1", "2", _iso(datetime.datetime(2026, 11, 15, 20, 30, tzinfo=ET)),
                        neutral=True, notes="NBA Cup Championship", score=(150, 50)))
    # Yesterday (Nov 20): exactly these teams played → B2B cases today.
    for h, a in (("24", "3"), ("25", "5"), ("7", "8")):
        events.append(final(h, a, _iso(datetime.datetime(2026, 11, 20, 19, 30, tzinfo=ET))))
    return events, strength


HISTORY, TRUE_STRENGTH = build_history()

# Today's slate (Saturday Nov 21) — the §8.3 cases. Tips in UTC.
SLATE = [
    dict(eid="401900001", home="1", away="2", tip="2026-11-22T00:00Z", final=(110, 105), ot=True),
    dict(eid="401900002", home="4", away="3", tip="2026-11-21T18:00Z", final=(101, 99)),   # 1 pm ET
    dict(eid="401900003", home="5", away="6", tip="2026-11-22T00:30Z", final=(120, 100)),
    dict(eid="401900004", home="7", away="8", tip="2026-11-22T01:00Z", final=(99, 104)),
    dict(eid="401900005", home="9", away="30", tip="2026-11-22T02:00Z", neutral=True,
         venue="Arena CDMX", final=(115, 110)),
    dict(eid="401900006", home="11", away="12", tip="2026-11-22T00:00Z", postponed=True,
         resched="2026-11-25T00:00Z", final=(100, 90)),
    dict(eid="401900007", home="13", away="14", tip="2026-11-22T03:00Z", season_type=1,
         final=(100, 95)),
    dict(eid="401900008", home="15", away="16", tip="2026-11-21T14:00Z", final=(108, 102)),  # 9 am ET → live
    dict(eid="401900009", home="17", away="18", tip="2026-11-22T00:00Z", final=(90, 95)),
    dict(eid="401900010", home="19", away="20", tip="2026-11-22T00:00Z", final=(103, 100)),
    dict(eid="401900011", home="21", away="22", tip="2026-11-22T01:30Z", neutral=True,
         notes="NBA Cup Championship", final=(112, 108)),
]
SUNDAY_SLATE = [
    dict(eid="401900021", home="23", away="24", tip="2026-11-22T20:00Z", final=(100, 99)),
    dict(eid="401900031", home="25", away="26", tip="2026-11-26T00:00Z", final=(104, 101)),  # Wed 7 pm
]
EIDS = {g["home"]: g["eid"] for g in SLATE}


def slate_event(g, now):
    tip = datetime.datetime.fromisoformat(g["tip"].replace("Z", "+00:00"))
    stype = g.get("season_type", 2)
    if g.get("postponed"):
        rs = datetime.datetime.fromisoformat(g["resched"].replace("Z", "+00:00"))
        if now >= rs + datetime.timedelta(hours=3):
            hs, as_ = g["final"]
            return espn_event(g["eid"], g["home"], g["away"], g["resched"], "STATUS_FINAL",
                              "post", True, 4, stype, hs=hs, as_=as_, detail="Final")
        return espn_event(g["eid"], g["home"], g["away"], g["tip"], "STATUS_POSTPONED",
                          "pre", False, 0, stype, detail="Postponed")
    if now >= tip + datetime.timedelta(hours=2, minutes=30):
        hs, as_ = g["final"]
        return espn_event(g["eid"], g["home"], g["away"], g["tip"], "STATUS_FINAL", "post",
                          True, 5 if g.get("ot") else 4, stype, g.get("neutral", False),
                          g.get("notes"), hs, as_, "Final", g.get("venue", "Mock Arena"))
    if now >= tip:
        return espn_event(g["eid"], g["home"], g["away"], g["tip"], "STATUS_IN_PROGRESS",
                          "in", False, 3, stype, g.get("neutral", False), g.get("notes"),
                          hs=60, as_=55, detail="3rd Qtr")
    return espn_event(g["eid"], g["home"], g["away"], g["tip"], "STATUS_SCHEDULED", "pre",
                      False, 0, stype, g.get("neutral", False), g.get("notes"),
                      venue=g.get("venue", "Mock Arena"))


def event_date_et(ev) -> str:
    dt = datetime.datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
    return dt.astimezone(ET).strftime("%Y-%m-%d")


# ── Mock Odds API ──────────────────────────────────────────────────────────────
def book(key, home, away, sp=None, ml=None):
    markets = []
    if ml:
        markets.append({"key": "h2h", "outcomes": [
            {"name": team_name(home), "price": ml[0]},
            {"name": team_name(away), "price": ml[1]}]})
    if sp:
        pt, ph, pa = sp
        markets.append({"key": "spreads", "outcomes": [
            {"name": team_name(home), "price": ph, "point": pt},
            {"name": team_name(away), "price": pa, "point": -pt}]})
    return {"key": key, "title": key, "markets": markets}


def odds_event(g, books):
    # The Odds API spells the Clippers "Los Angeles Clippers" — exercise the alias.
    def nm(tid):
        return "Los Angeles Clippers" if tid == "12" else team_name(tid)
    return {"id": "o" + g["eid"], "sport_key": "basketball_nba",
            "commence_time": g["tip"].replace("Z", ":00Z"),
            "home_team": nm(g["home"]), "away_team": nm(g["away"]), "bookmakers": books}


def default_books(h, a):
    return [book("pinnacle", h, a, (-4.5, -108, -112), (-190, 170)),
            book("fanduel", h, a, (-4.5, -110, -110), (-185, 160)),
            book("draftkings", h, a, (-4.5, -110, -110), (-190, 165)),
            book("betmgm", h, a, (-4.5, -110, -110), (-185, 160)),
            book("caesars", h, a, (-4.5, -110, -110), (-190, 165))]


def feed_for(g):
    h, a = g["home"], g["away"]
    if g["eid"] == "401900001":     # both markets GOOD on the home side
        return odds_event(g, [book("pinnacle", h, a, (-6.5, -110, -110), (-280, 230)),
                              book("fanduel", h, a, (-4.5, -110, -110), (-200, 170)),
                              book("draftkings", h, a, (-6.5, -110, -110), (-270, 220))])
    if g["eid"] == "401900008":     # in-play prices — must be filtered
        return odds_event(g, [book("pinnacle", h, a, (-25.5, -110, -110), (-5000, 2000)),
                              book("fanduel", h, a, (-25.5, -110, -110), (-5000, 2000))])
    if g["eid"] == "401900009":     # Pinnacle missing → consensus of 3
        return odds_event(g, [book("fanduel", h, a, (-4.5, -110, -110), (-185, 160)),
                              book("draftkings", h, a, (-4.5, -110, -110), (-190, 165)),
                              book("betmgm", h, a, (-5, -110, -110), (-200, 170)),
                              book("caesars", h, a, (-5.5, -110, -110), (-210, 175))])
    if g["eid"] == "401900010":     # whole-number spread
        return odds_event(g, [book("pinnacle", h, a, (-3, -110, -110), (-150, 130)),
                              book("fanduel", h, a, (-3, -110, -110), (-150, 130))])
    if g.get("postponed"):
        return None
    return odds_event(g, default_books(h, a))


# ── requests.get router ────────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code = payload, status
        self.headers = headers or {}
        self.text = json.dumps(payload)[:200]

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class Router:
    def __init__(self):
        self.range_broken = False
        self.calls = []

    def now(self):
        return datetime.datetime.fromisoformat(os.environ["NBA_APP_CLOCK"]).astimezone(UTC)

    def all_events(self):
        now = self.now()
        evs = list(HISTORY)
        evs += [slate_event(g, now) for g in SLATE]
        evs += [slate_event(g, now) for g in SUNDAY_SLATE]
        return evs

    def __call__(self, url, params=None, timeout=None, **kw):
        params = params or {}
        self.calls.append((url, dict(params)))
        if "the-odds-api.com" in url:
            feed = [feed_for(g) for g in SLATE + SUNDAY_SLATE]
            return FakeResp([f for f in feed if f], headers={"x-requests-remaining": "417"})
        if "/standings" in url:
            entries = [{"team": {"id": t, "displayName": team_name(t)},
                        "stats": [{"name": "avgPointsFor", "value": 112.0 + TRUE_STRENGTH[t] / 2},
                                  {"name": "avgPointsAgainst", "value": 112.0 - TRUE_STRENGTH[t] / 2},
                                  {"name": "gamesPlayed", "value": 82}]}
                       for t in TEAM_IDS]
            return FakeResp({"children": [{"name": "East", "standings": {"entries": entries[:15]}},
                                          {"name": "West", "standings": {"entries": entries[15:]}}]})
        if "/scoreboard" in url:
            d = str(params.get("dates", ""))
            if "-" in d:
                if self.range_broken:
                    return FakeResp({"events": []})
                a, b = d.split("-")
                lo, hi = f"{a[:4]}-{a[4:6]}-{a[6:]}", f"{b[:4]}-{b[4:6]}-{b[6:]}"
                evs = [e for e in self.all_events() if lo <= event_date_et(e) <= hi]
            else:
                day = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                evs = [e for e in self.all_events() if event_date_et(e) == day]
            return FakeResp({"events": evs})
        raise AssertionError(f"unexpected URL {url}")


# ── Harness plumbing ───────────────────────────────────────────────────────────
@pytest.fixture
def app_dir(tmp_path):
    for f in ("nba_app.py", "nba_teams.py"):
        shutil.copy(os.path.join(REPO, f), tmp_path / f)
    totals = {"teams": {t: {"team": team_name(t), "win_total": 41 + TRUE_STRENGTH[t] * 2.7}
                        for t in TEAM_IDS if t not in ("29", "30")}}
    totals["teams"]["29"] = {"team": team_name("29"), "win_total": None}
    (tmp_path / "nba_preseason_totals_2026.json").write_text(json.dumps(totals))
    return tmp_path


@pytest.fixture
def router(monkeypatch):
    r = Router()
    monkeypatch.setattr("requests.get", r)
    import streamlit as st
    st.cache_data.clear()
    return r


def run_app(app_dir, clock, monkeypatch, secrets=True):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    monkeypatch.setenv("NBA_APP_CLOCK", clock)
    st.cache_data.clear()
    at = AppTest.from_file(str(app_dir / "nba_app.py"), default_timeout=60)
    if secrets:
        at.secrets["odds"] = {"api_key": "test-key", "book": "fanduel", "reference": "pinnacle"}
    at.run()
    assert not at.exception, at.exception
    return at


def read_log(app_dir):
    p = app_dir / "nba_pick_log.json"
    return json.loads(p.read_text()) if p.exists() else []


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception, at.exception
    return at


def df_text(at):
    return "\n".join(d.value.to_string() for d in at.dataframe)


def all_text(at):
    return "\n".join(m.value for m in at.markdown) + "\n" + \
           "\n".join(c.value for c in at.caption) + "\n" + \
           "\n".join(i.value for i in at.info) + "\n" + \
           "\n".join(w.value for w in at.warning) + "\n" + \
           "\n".join(s.value for s in at.success) + "\n" + \
           "\n".join(e.value for e in at.error)


def load_core(app_dir):
    """Exec the pure half of nba_app.py (everything above the sidebar) in
    bare mode so the model/gate math is testable without a UI run."""
    src = (app_dir / "nba_app.py").read_text()
    head = src.split("# ── Sidebar")[0]
    ns = {"__file__": str(app_dir / "nba_app.py"), "__name__": "nba_core"}
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    exec(compile(head, str(app_dir / "nba_app.py"), "exec"), ns)
    return ns


# ══════════════════════════════════════════════════════════════════════════════
# Pure model / gate math
# ══════════════════════════════════════════════════════════════════════════════
def test_norm_ppf_matches_known_quantiles(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    assert abs(c["norm_ppf"](0.975) - 1.959964) < 1e-6
    assert abs(c["norm_ppf"](0.5)) < 1e-9
    assert abs(c["norm_ppf"](0.01) + 2.326348) < 1e-6
    for p in (0.001, 0.2, 0.5, 0.77, 0.999):
        assert abs(c["norm_cdf"](c["norm_ppf"](p)) - p) < 1e-7


def test_form_and_prior_weights(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    assert c["form_weight"](0) == 0.0 and c["form_weight"](15) == 0.0
    assert c["form_weight"](30) == 0.0            # share 0.5 > 0.30 → season avg already over-weights
    assert abs(c["form_weight"](60) - 0.05 / 0.75) < 1e-12
    assert c["prior_weight"](0) == 1.0
    assert abs(c["prior_weight"](6) - 0.6) < 1e-12
    assert c["prior_weight"](15) == 0.0 and c["prior_weight"](40) == 0.0
    # prior map: 55-win team → +5.19 pts
    pri = c["build_priors"]({"2": 55.0}, {"30": -8.0})
    assert abs(pri["2"]["margin"] - (55 - 41) / 2.7) < 1e-9
    assert abs(pri["30"]["margin"] - (-8.0 * 0.6)) < 1e-9 and pri["30"]["source"] != "win total"
    assert pri["1"]["margin"] == 0.0


def test_projection_hfa_rest_and_probabilities(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    M, hfa, rest = c["project_margin"](3.0, -1.0, False, False, False)
    assert (M, hfa, rest) == (6.5, 2.5, 0.0)
    assert c["project_margin"](3.0, -1.0, True, False, False)[0] == 4.0      # neutral: no HFA
    assert c["project_margin"](3.0, -1.0, False, False, True)[2] == 1.5      # away on B2B favors home
    assert c["project_margin"](3.0, -1.0, False, True, False)[2] == -1.5     # home on B2B
    assert c["project_margin"](3.0, -1.0, False, True, True)[2] == 0.0       # both → nets to 0
    assert abs(c["p_home_win"](0) - 0.5) < 1e-12
    assert abs(c["p_home_win"](12.5) - c["norm_cdf"](1.0)) < 1e-12
    # cover prob at the model's own line is exactly 0.5; same SIGMA for both
    assert abs(c["p_home_cover"](6.5, -6.5) - 0.5) < 1e-12
    assert abs(c["p_home_cover"](6.5, -4.5) - c["norm_cdf"](2 / 12.5)) < 1e-12
    assert c["tier_for"](0.66) == "High" and c["tier_for"](0.60) == "Moderate" \
        and c["tier_for"](0.55) == "Low"


def test_market_gate_numbers_and_verdicts(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    books = {"pinnacle": {"h2h": {"home": -280, "away": 230},
                          "spreads": {"home": (-6.5, -110), "away": (6.5, -110)}},
             "fanduel": {"h2h": {"home": -200, "away": 170},
                         "spreads": {"home": (-4.5, -110), "away": (4.5, -110)}}}
    ref = c["reference_lines"](books, "fanduel", "pinnacle")
    assert ref["ats"]["src"] == "pinnacle" and abs(ref["ats"]["fair_margin"] - 6.5) < 1e-9
    assert abs(ref["ml"]["fair_home"] - (280 / 380) / (280 / 380 + 100 / 330)) < 1e-9
    fd = {"spread": -4.5, "sp_home": -110, "sp_away": -110, "ml_home": -200, "ml_away": 170}
    mg = c["market_gate"](fd, ref)
    h = mg["ats"]["sides"]["home"]
    assert abs(h["fair"] - c["norm_cdf"](2.0 / 12.5)) < 1e-9
    assert abs(h["edge"] - (c["norm_cdf"](0.16) - 110 / 210)) < 1e-9
    assert h["verdict"] == "GOOD PRICE" and mg["ats"]["side"] == "home"
    assert mg["ats"]["sides"]["away"]["verdict"] == "STAY AWAY"
    m = mg["ml"]["sides"]["home"]
    assert abs(m["edge"] - (ref["ml"]["fair_home"] - 200 / 300)) < 1e-9
    assert m["verdict"] == "GOOD PRICE" and mg["ml"]["side"] == "home"
    pos = c["position_from_gate"](mg)
    assert pos["market"] == "ATS" and pos["side"] == "home" and pos["line"] == -4.5
    assert "ML also GOOD" in pos["note"]
    # THIN band: 0 ≤ edge < 1.5pp
    fd2 = dict(fd, spread=-5.5)
    mg2 = c["market_gate"](fd2, ref)
    e = mg2["ats"]["sides"]["home"]["edge"]
    assert 0 <= e < 0.015 and mg2["ats"]["sides"]["home"]["verdict"] == "THIN — NO BET"
    # Consensus fallback when Pinnacle is absent: mean of the others' M*.
    books2 = {"fanduel": books["fanduel"],
              "draftkings": {"spreads": {"home": (-4.5, -110), "away": (4.5, -110)}},
              "betmgm": {"spreads": {"home": (-5.0, -110), "away": (5.0, -110)}},
              "caesars": {"spreads": {"home": (-5.5, -110), "away": (5.5, -110)}}}
    ref2 = c["reference_lines"](books2, "fanduel", "pinnacle")
    assert ref2["ats"]["src"] == "consensus of 3" and abs(ref2["ats"]["fair_margin"] - 5.0) < 1e-9
    assert ref2["ml"]["src"] is None and c["market_gate"](fd, ref2)["ml"] is None
    # No reference at all → no market gate; paper gate still runs.
    ref3 = c["reference_lines"]({}, "fanduel", "pinnacle")
    assert c["market_gate"](fd, ref3) == {"ats": None, "ml": None}
    pg = c["paper_gate"](6.5, fd)
    assert pg["ats"]["side"] == "home" and abs(pg["ats"]["p"] - c["norm_cdf"](0.16)) < 1e-9
    assert pg["ats"]["verdict"] == "GOOD PRICE"                 # 56.4% vs 52.4% → +4pp
    assert pg["ml"]["side"] == "home" and pg["ml"]["verdict"] == "GOOD PRICE"  # 69.9% vs 66.7% → +3.2pp
    assert abs(pg["ml"]["cushion"] - (c["norm_cdf"](6.5 / 12.5) - 200 / 300)) < 1e-9


def test_grading_push_and_clv(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    assert c["grade_ats"]("home", -3.0, 103, 100) == "Push"
    assert c["grade_ats"]("away", 3.0, 103, 100) == "Push"
    assert c["grade_ats"]("home", -4.5, 110, 105) == "W"
    assert c["grade_ats"]("away", 4.5, 110, 105) == "L"
    assert c["grade_ml"]("home", 110, 105) == "W" and c["grade_ml"]("away", 110, 105) == "L"
    assert c["clv_ats_pts"](-4.5, -5.5) == 1.0 and c["clv_ats_pts"](4.5, 2.5) == 2.0
    assert c["clv_ml_pp"](-200, -220) == pytest.approx((220 / 320 - 200 / 300) * 100, abs=0.01)
    assert c["units_for"]("W", -110) == pytest.approx(100 / 110)
    assert c["units_for"]("L", -110) == -1.0 and c["units_for"]("Push", -110) == 0.0
    # grade_row end to end, with a close snapshot
    row = {"espn_event_id": "1", "home_id": "1", "away_id": "2", "home": team_name("1"),
           "away": team_name("2"), "market_ats_side": team_name("1"), "market_ats_line": -4.5,
           "market_ats_price": -110, "market_ml_side": team_name("1"), "market_ml_price": -200,
           "paper_ats_side": team_name("2"), "paper_ats_line": 4.5, "paper_ats_price": -110,
           "paper_ml_side": team_name("1"), "paper_ml_price": -200,
           "bet_market": "ATS", "bet_side": team_name("1"), "bet_line": -4.5,
           "bet_price": -110, "stake": 50.0}
    gm = {"completed": True, "home_score": 110, "away_score": 105, "ot": True}
    closes = {"1": {"quotes": {"fanduel": {"h2h": {"home": -220, "away": 185},
                                           "spreads": {"home": [-5.5, -110], "away": [5.5, -110]}}}}}
    assert c["grade_row"](row, gm, closes, "fanduel")
    assert row["ot"] is True and row["result_ats"] == "W" and row["result_ml"] == "W"
    assert row["result_paper_ats"] == "L" and row["clv_paper_ats_pts"] == -1.0
    assert row["clv_ats_pts"] == 1.0 and abs(row["clv_ml_pp"] - 2.08) < 0.01
    assert row["result_bet"] == "W" and row["pnl"] == pytest.approx(50 * 100 / 110, abs=0.01)
    assert row["bet"] is True and row["graded_at"]


def test_history_excludes_cup_final_and_preseason(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    parsed = [c["_parse_event"](e) for e in HISTORY]
    assert all(parsed)
    cup = [g for g in parsed if g["cup_final"]]
    assert len(cup) == 1 and cup[0]["neutral"] and cup[0]["regular"]
    ots = [g for g in parsed if g["ot"]]
    assert len(ots) == 1 and ots[0]["period"] == 5
    season_log, played, records = c["build_gamelogs"](parsed)
    n1 = sum(1 for g in parsed if g["regular"] and not g["cup_final"]
             and "1" in (g["home_id"], g["away_id"]))
    assert len(season_log["1"]) == n1
    assert not any(pf == 150 for _, pf, _ in season_log["1"])     # cup final excluded
    assert not any(g["preseason"] for g in parsed
                   if g["event_id"] in {x["event_id"] for x in parsed if x["regular"]})
    n_pre_teams = {g["home_id"] for g in parsed if g["preseason"]}
    assert n_pre_teams                                            # preseason exists...
    assert sum(len(v) for v in season_log.values()) // 2 == \
        sum(1 for g in parsed if g["regular"] and not g["cup_final"])   # ...and is not in the log
    assert ("3", "2026-11-20") in played and ("5", "2026-11-20") in played
    assert ("1", "2026-11-20") not in played
    strengths = c["team_strengths"](season_log, c["build_priors"]({}, {}))
    assert strengths["30"]["n"] < 15 < strengths["1"]["n"]
    assert strengths["30"]["w_prior"] > 0 == strengths["1"]["w_prior"]
    # A 1 pm weekend tip parses to the right ET label and date.
    g = c["_parse_event"](slate_event(SLATE[1], datetime.datetime(2026, 11, 21, 15, tzinfo=UTC)))
    assert g["tip_label"] == "Sat 1:00 PM ET" and g["date_et"] == SAT


def test_odds_feed_parse_and_inplay_filter(app_dir, monkeypatch):
    monkeypatch.setenv("NBA_APP_CLOCK", CLOCK_SAT)
    c = load_core(app_dir)
    ev = c["parse_odds_event"](feed_for(SLATE[0]))
    assert ev["home_id"] == "1" and ev["away_id"] == "2"
    assert ev["books"]["fanduel"]["spreads"]["home"] == (-4.5, -110)
    assert ev["books"]["pinnacle"]["h2h"] == {"home": -280, "away": 230}
    live = c["parse_odds_event"](feed_for(SLATE[7]))
    g8 = c["_parse_event"](slate_event(SLATE[7], c["now_utc"]()))
    assert c["match_odds_event"]([live], g8) == (None, "live")
    g1 = c["_parse_event"](slate_event(SLATE[0], c["now_utc"]()))
    assert c["match_odds_event"]([ev], g1)[1] == "ok"
    assert c["match_odds_event"]([ev], g8) == (None, "unmatched")


# ══════════════════════════════════════════════════════════════════════════════
# AppTest — the slate on Saturday
# ══════════════════════════════════════════════════════════════════════════════
def test_saturday_slate_gates_and_log(app_dir, router, monkeypatch):
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    text = all_text(at)
    assert "Win totals for 28/30" in text                    # 29 null, 30 missing → fallback
    assert "417 credits remaining" in text                    # burn is visible
    assert "9/10 games priced" in text                        # G6 postponed excluded; G8 live blank
    # Prefill from the feed (session state set before the widgets)
    assert at.session_state[f"nba1_sp_{EIDS['1']}"] == -4.5
    assert at.session_state[f"nba1_mlh_{EIDS['1']}"] == -200
    assert at.session_state[f"nba1_sp_{EIDS['15']}"] == 0.0     # in-play: left blank
    assert "Game underway" in text
    assert "NBA CUP FINAL" in text and "Neutral site" in text
    assert "PRESEASON" in text and "back-to-back" in text
    assert "Sat 1:00 PM ET" in text
    assert "POSITION: ATS" in text and "ML also GOOD" in text
    assert "consensus of 3" in text

    click(at, "nba1_log")
    log = read_log(app_dir)
    by = {r["espn_event_id"]: r for r in log}
    assert len(log) == 10 and EIDS["13"] not in by             # preseason never logged
    assert all(r["version"] == "nba-v1.0" for r in log)
    g1 = by[EIDS["1"]]
    assert g1["verdict_market_ats"] == "GOOD PRICE" and g1["verdict_market_ml"] == "GOOD PRICE"
    assert g1["market_ats_side"] == team_name("1") == g1["market_ml_side"]
    assert g1["bet_market"] == "ATS" and g1["bet_line"] == -4.5 and g1["bet_price"] == -110
    assert "ML also GOOD" in g1["position_note"] and g1["stake"] == 0 and g1["bet"] is False
    assert g1["fd_source"] == "feed" and g1["ref_book"] == "pinnacle"
    assert g1["fair_margin"] == 6.5 and g1["edge_ats_home"] == pytest.approx(3.97, abs=0.02)
    assert g1["paper_counts"] is True and g1["ot"] is None and g1["graded_at"] is None
    g2 = by[EIDS["4"]]
    assert g2["away_b2b"] is True and g2["home_b2b"] is False and g2["rest_adj"] == 1.5
    assert g2["tip_et"].startswith("2026-11-21T13:00")
    g3 = by[EIDS["5"]]
    assert g3["home_b2b"] is True and g3["away_b2b"] is False and g3["rest_adj"] == -1.5
    g4 = by[EIDS["7"]]
    assert g4["home_b2b"] and g4["away_b2b"] and g4["rest_adj"] == 0.0
    g5 = by[EIDS["9"]]
    assert g5["neutral_site"] is True and g5["hfa"] == 0.0 and g5["paper_counts"] is False
    g6 = by[EIDS["11"]]
    assert g6["postponed"] is True and g6["fd_source"] == "none" and g6["verdict_market_ats"] is None
    g8 = by[EIDS["15"]]
    assert g8["in_play_at_log"] is True and g8["fd_source"] == "none"
    assert g8["verdict_market_ats"] is None and g8["verdict_paper_ats"] is None
    g9 = by[EIDS["17"]]
    assert g9["ref_book"] == "consensus of 3" and g9["fair_margin"] == pytest.approx(5.0)
    g10 = by[EIDS["19"]]
    assert g10["fd_spread"] == -3.0 and g10["verdict_market_ats"] == "STAY AWAY"
    g11 = by[EIDS["21"]]
    assert g11["cup_final"] is True and g11["neutral_site"] is True
    # model margin = strengths + hfa + rest, straight from the row
    for r in log:
        assert isinstance(r["model_margin"], float) and isinstance(r["p_home_win"], float)
    # Idempotent: a second click adds nothing.
    click(at, "nba1_log")
    assert len(read_log(app_dir)) == 10
    assert "already logged" in all_text(at)


def test_manual_override_flags_source(app_dir, router, monkeypatch):
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    eid = EIDS["1"]
    at.number_input(key=f"nba1_sp_{eid}").set_value(-3.5)
    at.run()
    assert not at.exception
    assert at.session_state[f"nba1_manual_{eid}"] is True
    click(at, "nba1_log")
    g1 = {r["espn_event_id"]: r for r in read_log(app_dir)}[eid]
    assert g1["fd_source"] == "manual" and g1["fd_spread"] == -3.5
    assert g1["ref_book"] == "pinnacle"                       # reference still from the feed
    assert g1["edge_ats_home"] > 3.98                          # a better number → bigger edge


def test_no_feed_means_no_market_gate(app_dir, router, monkeypatch):
    at = run_app(app_dir, CLOCK_SAT, monkeypatch, secrets=False)
    assert "No odds feed configured" in all_text(at)
    click(at, "nba1_log")
    log = read_log(app_dir)
    assert len(log) == 10
    assert all(r["verdict_market_ats"] is None and r["bet_market"] is None for r in log)
    assert all(r["fd_source"] == "none" for r in log)
    assert all(r["paper_ml_side"] for r in log)                # model still picks


def test_day_with_no_games(app_dir, router, monkeypatch):
    at = run_app(app_dir, CLOCK_OFF, monkeypatch)
    assert "No NBA games scheduled for today" in all_text(at)


def test_range_query_fallback_to_per_day(app_dir, router, monkeypatch):
    router.range_broken = True
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    text = all_text(at)
    assert "regular-season games" in text and "No 2026-27 regular-season games" not in text
    per_day = [p for u, p in router.calls if "/scoreboard" in u and "-" not in str(p.get("dates", ""))]
    assert len(per_day) > 30                                   # October + November days


# ══════════════════════════════════════════════════════════════════════════════
# AppTest — auto-grade the next morning, then the postponed replay
# ══════════════════════════════════════════════════════════════════════════════
def write_closes(app_dir):
    d = app_dir / "odds_closes" / "nba"
    d.mkdir(parents=True)
    closes = {
        EIDS["1"]: {"quotes": {"fanduel": {"h2h": {"home": -220, "away": 185},
                                           "spreads": {"home": [-5.5, -110], "away": [5.5, -110]}}}},
        EIDS["19"]: {"quotes": {"fanduel": {"h2h": {"home": -150, "away": 130},
                                            "spreads": {"home": [-3.0, -110], "away": [3.0, -110]}}}},
    }
    (d / f"{SAT}.json").write_text(json.dumps(closes))


def test_autograde_ot_push_postponed_and_clv(app_dir, router, monkeypatch):
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    click(at, "nba1_log")
    # Stake the one position (STAKE_UNITS is 0 in code → refused, nothing staked).
    click(at, "nba1_stake")
    assert "STAKE_UNITS / SEASON_BUDGET are 0" in all_text(at)
    assert all(r["stake"] == 0 for r in read_log(app_dir))
    write_closes(app_dir)

    at = run_app(app_dir, CLOCK_SUN, monkeypatch)
    assert "1/1 games priced" in all_text(at)                  # Sunday's slate is live
    click(at, "nba1_grade")
    log = read_log(app_dir)
    by = {r["espn_event_id"]: r for r in log}
    g1 = by[EIDS["1"]]
    assert g1["final_home"] == 110 and g1["final_away"] == 105 and g1["ot"] is True
    assert g1["result_ats"] == "W" and g1["result_ml"] == "W"
    assert g1["close_spread"] == -5.5 and g1["clv_ats_pts"] == 1.0
    assert g1["close_ml_home"] == -220 and abs(g1["clv_ml_pp"] - 2.08) < 0.01
    assert g1["result_bet"] == "W" and g1["pnl"] is None       # position logged, never staked
    assert g1["graded_at"]
    g10 = by[EIDS["19"]]
    assert g10["result_ats"] == "Push" and g10["result_paper_ats"] == "Push"
    assert g10["clv_ats_pts"] == 0.0                            # CLV still recorded on a push
    g6 = by[EIDS["11"]]
    assert g6["postponed"] is True and not g6["graded_at"]
    g8 = by[EIDS["15"]]
    assert g8["graded_at"] and g8["result_paper_ml"] in ("W", "L") and g8["result_ats"] == ""
    g5 = by[EIDS["9"]]
    assert g5["graded_at"] and g5["result_ats"] in ("W", "L")
    n_graded = sum(1 for r in log if r["graded_at"])
    assert n_graded == 9                                        # all but the postponed one
    text = all_text(at)
    tables = df_text(at)
    assert "market_ats · GOOD PRICE" in tables and "paper_ats · GOOD · counts" in tables
    assert "Auto-graded 9" in text
    # Graded rows are immutable: grading again changes nothing.
    snapshot = json.dumps(log, sort_keys=True)
    click(at, "nba1_grade")
    assert json.dumps(read_log(app_dir), sort_keys=True) == snapshot

    # Wednesday: the postponed game replayed Tuesday under the same event ID.
    at = run_app(app_dir, CLOCK_WED, monkeypatch)
    click(at, "nba1_grade")
    g6 = {r["espn_event_id"]: r for r in read_log(app_dir)}[EIDS["11"]]
    assert g6["graded_at"] and g6["postponed"] is False and g6["rescheduled_to"] == "2026-11-24"
    assert g6["final_home"] == 100 and g6["result_paper_ml"] in ("W", "L")


def test_superseded_version_rows_are_replaced(app_dir, router, monkeypatch):
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    click(at, "nba1_log")
    log = read_log(app_dir)
    # Pretend two rows came from an older frozen model: one ungraded, one graded.
    log[0]["version"] = "nba-v0.9"
    log[1]["version"] = "nba-v0.9"; log[1]["graded_at"] = "2026-11-21T09:00"
    (app_dir / "nba_pick_log.json").write_text(json.dumps(log))
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    click(at, "nba1_log")
    log2 = read_log(app_dir)
    assert "Replaced 1 ungraded row" in all_text(at)
    assert sum(1 for r in log2 if r["version"] == "nba-v0.9") == 1          # the graded one stays
    assert sum(1 for r in log2 if r["version"] == "nba-v1.0") == 10          # replaced + the rest
    assert len(log2) == 11


# ══════════════════════════════════════════════════════════════════════════════
# Closes cron — the spending decision
# ══════════════════════════════════════════════════════════════════════════════
def test_cron_only_spends_inside_the_window(monkeypatch):
    import snapshot_closes_nba as cron
    tip7 = datetime.datetime(2026, 11, 22, 0, 0, tzinfo=UTC)      # 7:00 pm ET
    games = [{"event_id": "a", "tip": tip7},
             {"event_id": "b", "tip": tip7 + datetime.timedelta(minutes=10)},
             {"event_id": "c", "tip": tip7 + datetime.timedelta(minutes=30)},
             {"event_id": "d", "tip": tip7 + datetime.timedelta(hours=3)}]
    at_617 = tip7 - datetime.timedelta(minutes=43)
    at_647 = tip7 - datetime.timedelta(minutes=13)
    at_717 = tip7 + datetime.timedelta(minutes=17)
    assert cron.games_needing_close(games, {}, at_617) == []           # nothing within 35 min
    need = cron.games_needing_close(games, {}, at_647)
    assert [g["event_id"] for g in need] == ["a", "b"]                 # c is 43 min out
    captured = {"a": {"quoted_at": at_647.isoformat()}, "b": {"quoted_at": at_647.isoformat()}}
    need = cron.games_needing_close(games, captured, at_717)
    assert [g["event_id"] for g in need] == ["c"]                      # a/b captured, a started
    stale = {"c": {"quoted_at": (tip7 - datetime.timedelta(hours=2)).isoformat()}}
    assert [g["event_id"] for g in cron.games_needing_close(games, stale, at_717)] == ["c"]
    # A 1 pm weekend tip: 12:47 pm run captures it, 12:17 doesn't.
    tip1 = datetime.datetime(2026, 11, 21, 18, 0, tzinfo=UTC)
    g1 = [{"event_id": "m", "tip": tip1}]
    assert cron.games_needing_close(g1, {}, tip1 - datetime.timedelta(minutes=43)) == []
    assert len(cron.games_needing_close(g1, {}, tip1 - datetime.timedelta(minutes=13))) == 1


def test_cron_merge_keys_by_event_id_and_keeps_started_games():
    import snapshot_closes_nba as cron
    now = datetime.datetime(2026, 11, 21, 23, 47, tzinfo=UTC)
    games = [{"event_id": "401900001", "home_id": "1", "away_id": "2", "home": team_name("1"),
              "away": team_name("2"), "tip": datetime.datetime(2026, 11, 22, 0, 0, tzinfo=UTC)},
             {"event_id": "401900008", "home_id": "15", "away_id": "16", "home": team_name("15"),
              "away": team_name("16"), "tip": datetime.datetime(2026, 11, 21, 14, 0, tzinfo=UTC)}]
    feed = [cron.parse_feed_event(feed_for(SLATE[0])), cron.parse_feed_event(feed_for(SLATE[7]))]
    existing = {"401900008": {"quoted_at": "2026-11-21T13:47:00+00:00",
                              "quotes": {"fanduel": {"h2h": {"home": -150, "away": 130}}}}}
    merged, n = cron.merge_snapshot(existing, feed, games, now)
    assert n == 1 and set(merged) == {"401900001", "401900008"}
    assert merged["401900001"]["quotes"]["fanduel"]["spreads"]["home"] == [-4.5, -110]
    assert merged["401900008"]["quotes"]["fanduel"]["h2h"]["home"] == -150   # started: untouched


def test_cron_main_respects_reserve_and_window(tmp_path, monkeypatch, router):
    import snapshot_closes_nba as cron
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", "k")
    # 6:47 pm ET Saturday: G1/G9/G10 (7:00), G3 (7:30 → 43 min, no) …
    monkeypatch.setenv("NBA_APP_CLOCK", "2026-11-21T18:47:00-05:00")
    assert cron.main() == 0
    out = json.loads((tmp_path / "odds_closes/nba" / f"{SAT}.json").read_text())
    assert EIDS["1"] in out and EIDS["17"] in out and EIDS["19"] in out
    assert EIDS["5"] in out                    # 7:30 game refreshed too (feed had it, still upcoming)
    assert EIDS["15"] not in out               # 9 am game: started, never quoted
    odds_calls = [u for u, _ in router.calls if "the-odds-api" in u]
    assert len(odds_calls) == 1
    state = json.loads((tmp_path / "odds_closes/nba/_credits.json").read_text())
    assert state["remaining"] == 417
    # Same slot again → captured → no call.
    assert cron.main() == 0
    assert len([u for u, _ in router.calls if "the-odds-api" in u]) == 1
    # Reserve: pretend 30 credits left → 7:17 run (7:30 tip within window) must skip.
    (tmp_path / "odds_closes/nba/_credits.json").write_text(json.dumps({"remaining": 30}))
    monkeypatch.setenv("NBA_APP_CLOCK", "2026-11-21T19:17:00-05:00")
    assert cron.main() == 0
    assert len([u for u, _ in router.calls if "the-odds-api" in u]) == 1
    # Off-hours run: free.
    monkeypatch.setenv("NBA_APP_CLOCK", "2026-11-21T11:17:00-05:00")
    (tmp_path / "odds_closes/nba/_credits.json").write_text(json.dumps({"remaining": 400}))
    assert cron.main() == 0
    assert len([u for u, _ in router.calls if "the-odds-api" in u]) == 1


def test_espn_unreachable_degrades_without_exception(app_dir, monkeypatch):
    """The build container can't reach ESPN (egress policy) — the app must
    say so and stop, never trace back. Also exercises the per-day fallback
    when every request fails."""
    import streamlit as st
    st.cache_data.clear()

    def down(url, params=None, timeout=None, **kw):
        raise ConnectionError("CONNECT tunnel failed, response 403")
    monkeypatch.setattr("requests.get", down)
    at = run_app(app_dir, CLOCK_SAT, monkeypatch)
    text = all_text(at)
    assert "ESPN season pull failed" in text or "Could not load the schedule" in text
    assert "Could not load the schedule" in text

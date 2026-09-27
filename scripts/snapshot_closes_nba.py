#!/usr/bin/env python3
"""
snapshot_closes_nba.py — capture NBA closing lines on a credit budget.

Sibling of snapshot_closes.py (MLB). A separate script, not a parameter,
because the credit-saving mechanics below would not leave the MLB path
byte-identical (spec §5). The MLB script and workflow are untouched.

Run on a schedule (GitHub Actions, every 30 min at :17 and :47 across a
UTC window wide enough for 12:30 pm–11:00 pm ET in both EDT and EST). The
cron string never decides anything: THIS script does, on the ET clock —

  1. Pull the ET date's scoreboard from ESPN (free) → distinct tip times.
  2. Call the Odds API ONLY IF some game tips within the next 35 minutes
     AND that game's close hasn't been captured yet (a quote taken inside
     the 35-minute pre-tip window counts as captured). Otherwise exit
     without spending a credit.
  3. Snapshot = h2h + spreads, all books in ODDS_BOOKS → 2 credits. Written
     to odds_closes/nba/<ET date>.json keyed by ESPN EVENT ID; the last
     pre-tip quote for each game is the close (~13–23 min pre-tip on this
     cadence; late injury moves may be missed — accepted for the free tier).
  4. Hard reserve: if the last known x-requests-remaining < 40, skip and
     warn, so the app's own prefill stays alive at month end.

Budget math (spec §5): ~5 tip slots on a weeknight × 2 = 10 credits,
weekends ~8 × 2 = 16 → ≈ 300/month plus the app's prefill, under 350.

Env: ODDS_API_KEY (repo Actions secret, shared with the MLB workflow).
Imports nba_teams.py (fetched beside this script by the workflow).
"""
import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from nba_teams import team_id_for, team_name  # noqa: E402

ODDS_API     = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ODDS_MARKETS = "h2h,spreads"
ODDS_BOOKS   = "fanduel,pinnacle,draftkings,betmgm,caesars"
ESPN         = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CLOSES_DIR   = "odds_closes/nba"
STATE_FILE   = f"{CLOSES_DIR}/_credits.json"
WINDOW_MIN   = 35      # call only if a game tips within this many minutes
RESERVE      = 40      # never spend below this many remaining credits
ET  = ZoneInfo("America/New_York")
UTC = datetime.timezone.utc


def now_et() -> datetime.datetime:
    pin = os.environ.get("NBA_APP_CLOCK", "").strip()     # test seam only
    if pin:
        dt = datetime.datetime.fromisoformat(pin)
        return (dt if dt.tzinfo else dt.replace(tzinfo=ET)).astimezone(ET)
    return datetime.datetime.now(ET)


def parse_utc(iso: str) -> datetime.datetime | None:
    try:
        dt = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC)
    except Exception:
        return None


def espn_slate(date_iso: str) -> list:
    """[{event_id, home_id, away_id, home, away, tip (aware UTC), date_et}]
    for every event ESPN lists on that ET date, whatever its status."""
    r = requests.get(f"{ESPN}/scoreboard",
                     params={"dates": date_iso.replace("-", ""), "limit": 100},
                     timeout=20)
    r.raise_for_status()
    out = []
    for ev in r.json().get("events", []) or []:
        try:
            comp = ev["competitions"][0]
            sname = str(((ev.get("status") or {}).get("type") or {}).get("name", "")).upper()
            if sname in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED"):
                continue          # no tip → never a slot worth a credit
            sides = {c.get("homeAway"): str(c["team"]["id"])
                     for c in comp.get("competitors", [])}
            tip = parse_utc(ev.get("date"))
            if not tip or "home" not in sides or "away" not in sides:
                continue
            out.append({"event_id": str(ev["id"]),
                        "home_id": sides["home"], "away_id": sides["away"],
                        "home": team_name(sides["home"]),
                        "away": team_name(sides["away"]),
                        "tip": tip,
                        "date_et": tip.astimezone(ET).strftime("%Y-%m-%d")})
        except Exception:
            continue
    return out


def games_needing_close(games: list, existing: dict, now: datetime.datetime,
                        window_min: int = WINDOW_MIN) -> list:
    """Games that tip within the window AND have no quote taken inside the
    window yet. Pure — the cron's whole spending decision."""
    need = []
    win = datetime.timedelta(minutes=window_min)
    for g in games:
        if not (now < g["tip"] <= now + win):
            continue
        prev = (existing.get(g["event_id"]) or {}).get("quoted_at")
        prev_dt = parse_utc(prev) if prev else None
        if prev_dt is not None and prev_dt >= g["tip"] - win:
            continue           # this slot's close is already on file
        need.append(g)
    return need


def parse_feed_event(ev: dict) -> dict | None:
    """One Odds API event → {home_id, away_id, commence, quotes}. Same
    shape the app reads back (grade_row): quotes[book] = {"h2h": {"home":
    px, "away": px}, "spreads": {"home": [pt, px], "away": [pt, px]}}."""
    hid, aid = team_id_for(ev.get("home_team")), team_id_for(ev.get("away_team"))
    if not hid or not aid:
        return None
    quotes = {}
    for bk in ev.get("bookmakers", []) or []:
        q = {}
        for mk in bk.get("markets", []) or []:
            outs = mk.get("outcomes", []) or []
            if mk.get("key") == "h2h":
                d = {team_id_for(o.get("name")): int(o["price"]) for o in outs}
                if hid in d and aid in d:
                    q["h2h"] = {"home": d[hid], "away": d[aid]}
            elif mk.get("key") == "spreads":
                d = {team_id_for(o.get("name")): [float(o["point"]), int(o["price"])]
                     for o in outs if o.get("point") is not None}
                if hid in d and aid in d:
                    q["spreads"] = {"home": d[hid], "away": d[aid]}
        if q:
            quotes[bk.get("key", "")] = q
    return {"home_id": hid, "away_id": aid,
            "commence": ev.get("commence_time", ""), "quotes": quotes}


def merge_snapshot(existing: dict, feed: list, games: list,
                   now: datetime.datetime) -> tuple[dict, int]:
    """Overwrite the quote of every game still upcoming (commence > now)
    with the feed's current prices, keyed by ESPN event ID. Started games
    keep their last quote — that IS the close. Returns (merged, n_refreshed)."""
    by_teams = {}
    for g in games:
        by_teams.setdefault((g["home_id"], g["away_id"]), []).append(g)
    out = dict(existing)
    refreshed = 0
    for fe in feed:
        c = parse_utc(fe["commence"])
        if c is None or c <= now:
            continue
        cands = by_teams.get((fe["home_id"], fe["away_id"]), [])
        cands = [g for g in cands if abs((g["tip"] - c).total_seconds()) <= 6 * 3600]
        if not cands:
            continue
        g = min(cands, key=lambda x: abs((x["tip"] - c).total_seconds()))
        out[g["event_id"]] = {
            "home_id": g["home_id"], "away_id": g["away_id"],
            "home": g["home"], "away": g["away"],
            "commence": fe["commence"], "tip_espn": g["tip"].isoformat(),
            "quoted_at": now.astimezone(UTC).isoformat(), "quotes": fe["quotes"]}
        refreshed += 1
    return out, refreshed


def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def main() -> int:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ODDS_API_KEY missing")
        return 1
    now = now_et()
    date_iso = now.strftime("%Y-%m-%d")
    games = espn_slate(date_iso)
    games = [g for g in games if g["date_et"] == date_iso]
    print(f"{date_iso} ET {now:%H:%M}: {len(games)} games on ESPN's slate")
    if not games:
        return 0
    path = f"{CLOSES_DIR}/{date_iso}.json"
    existing = load_json(path, {})
    need = games_needing_close(games, existing, now.astimezone(UTC))
    if not need:
        print("no tip within the window needing a close — no credit spent")
        return 0
    state = load_json(STATE_FILE, {})
    remaining = state.get("remaining")
    if isinstance(remaining, (int, float)) and remaining < RESERVE:
        print(f"WARNING: only {remaining} credits remaining (< {RESERVE} reserve) "
              f"— skipping close for {[g['event_id'] for g in need]}")
        return 0
    print(f"tip within {WINDOW_MIN} min for {len(need)} game(s): "
          f"{[g['home'] + ' v ' + g['away'] for g in need]} → calling Odds API")
    r = requests.get(ODDS_API, params={"apiKey": key, "markets": ODDS_MARKETS,
                                       "oddsFormat": "american",
                                       "bookmakers": ODDS_BOOKS}, timeout=20)
    r.raise_for_status()
    rem = r.headers.get("x-requests-remaining")
    print(f"credits remaining: {rem}")
    os.makedirs(CLOSES_DIR, exist_ok=True)
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"remaining": int(float(rem)) if rem is not None else None,
                       "checked_at": now.astimezone(UTC).isoformat()}, f, indent=1)
    except Exception:
        pass
    feed = [e for e in (parse_feed_event(ev) for ev in r.json()) if e]
    merged, n = merge_snapshot(existing, feed, games, now.astimezone(UTC))
    with open(path, "w") as f:
        json.dump(merged, f, indent=1, sort_keys=True)
    print(f"{path}: {n} upcoming refreshed, {len(merged)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())

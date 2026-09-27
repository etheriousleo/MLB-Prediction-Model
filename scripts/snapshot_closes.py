#!/usr/bin/env python3
"""
snapshot_closes.py — capture closing lines automatically.

Run on a schedule (GitHub Actions cron, every 30 min during game hours).
Each run fetches current MLB moneylines for several books and, for every
game that has NOT started yet, overwrites that game's quote in
odds_closes/<ET date>.json. Because the file keeps being overwritten until
first pitch and never after, the quote left standing IS the closing line.
The app's Auto-grade reads this file from the data branch to fill `closing`.

Env: ODDS_API_KEY (repo Actions secret). Cost: 1 credit per run.
"""
import json
import os
import sys
import datetime
from zoneinfo import ZoneInfo

import requests

ODDS_API   = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
ODDS_BOOKS = "fanduel,pinnacle,draftkings,betmgm,caesars,betrivers"
ET = ZoneInfo("America/New_York")


def main():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ODDS_API_KEY missing"); sys.exit(1)
    r = requests.get(ODDS_API, params={"apiKey": key, "markets": "h2h",
                                       "oddsFormat": "american",
                                       "bookmakers": ODDS_BOOKS}, timeout=20)
    r.raise_for_status()
    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"credits remaining: {r.headers.get('x-requests-remaining')}")

    by_date = {}
    for ev in r.json():
        commence = datetime.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if commence <= now:
            continue                     # started: leave the last quote standing
        quotes = {}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "h2h":
                    quotes[bk["key"]] = {o["name"]: int(o["price"]) for o in mk["outcomes"]}
        d = commence.astimezone(ET).strftime("%Y-%m-%d")   # the app's ET game date
        by_date.setdefault(d, []).append({
            "home": ev["home_team"], "away": ev["away_team"],
            "commence": ev["commence_time"], "quoted_at": now.isoformat(),
            "quotes": quotes})

    os.makedirs("odds_closes", exist_ok=True)
    for d, events in by_date.items():
        path = f"odds_closes/{d}.json"
        existing = json.load(open(path)) if os.path.exists(path) else []
        keep = {(e["home"], e["away"], e["commence"]): e for e in existing}
        for e in events:                 # overwrite only games still upcoming
            keep[(e["home"], e["away"], e["commence"])] = e
        json.dump(sorted(keep.values(), key=lambda e: e["commence"]),
                  open(path, "w"), indent=1)
        print(f"{path}: {len(events)} upcoming refreshed, {len(keep)} total")


if __name__ == "__main__":
    main()

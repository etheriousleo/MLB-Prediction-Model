# MLB Prediction Model — working agreement for Claude Code

## What this is
Streamlit app (`mlb_app.py`) deployed on Streamlit Cloud from `main`. The
pick log, daily snapshots and closing-line snapshots live on the `data`
branch (`pick_log.json`, `snapshots/`, `odds_closes/`), written by the app
and by the GitHub Actions cron via the GitHub Contents API using secrets in
Streamlit Cloud and repo Actions secrets. Never commit tokens. Never commit
data to `main` (commits to `main` reboot the app).

## Status: 2026 SEASON CLOSED (2026-10-29). NO MLB CHANGES UNTIL THE JANUARY DECISION.
- `MODEL_VERSION = v4.2-market-gate-2026-09-27`. Final log: 726 rows.
- Binding records in the repo: `season_verdict_2026.md` (regular season,
  pre-registered) and `season_verdict_2026_postseason_addendum.md`.
- Regular-season verdict: the public-stat model carries no information the
  market lacks (λ*=0; market Brier .2362 vs model .2422; calibrated gate
  still −8.9%/bet). Postseason addendum: the FanDuel-vs-Pinnacle moneyline
  gate is structurally inert (0 feed-price fires in 45; median gap −2.1pp =
  the vig); model picks had a hot 52-game month in outcomes but −0.96pp CLV.
- The suite runs as data collection only — no wagers; do not nudge toward
  betting in any app.
- The MLB closing-line cron is DISABLED for the offseason (it burns Odds API
  credits shared with the NBA app). Re-enable in late March 2027.

## January decision (with the CFB post-bowls verdict in hand)
Three possible paths for MLB 2027; choose one, do not blend:
1. v5 model adding inputs the market prices late — confirmed lineups and
   bullpen usage from the MLB API — plus the calibration map (Platt slope
   1.81 fit on 2026, display only), forward-tested from Opening Day.
2. Market-research tool only: best price across all books the feed pulls
   vs Pinnacle; no model in the loop.
3. Shelve MLB; focus on the sports whose markets are softer (CFB verdict).
Optional offseason analysis: replay of 2026 snapshots with form weight at
0 / 15 / 35 (double-count fixed) — the only way to settle the form question
offline. Form diagnostic threshold → 1pp if it stays.

## Rules that exist because of real bugs
- All dates via `now_et()` / `today_et()`. Never naive `datetime` calls.
- Game identity is `matchup_key(game)` (doubleheaders get "(G2)").
- Every external team name passes through `canon_team()` (Athletics rename).
- Odds feed: exclude started games (live prices); pre-fill never overwrites a
  user-typed price; a row with an override or a stake is a taken price and is
  never re-synced.
- Budget guard is a LOSS LIMIT (limit + settled P/L − at risk), not a cap on
  stakes. Escape `$` in Streamlit markdown.
- Closing backfill runs for every row missing a closing, graded or not; never
  cache an empty snapshot.
- Before pushing: `python -m py_compile mlb_app.py`, local `streamlit run`,
  tracker loads from the data branch.

## Analysis tooling
- `log_integrity_check.py pick_log.json` — outcome-free hygiene; run first.
- `season_analysis.py pick_log.json` — the 2026 pre-registered workup;
  criteria LOCKED. For 2027, copy it, set a new lock date before the data
  is complete, never edit the criteria afterward.
- Method: freeze → log every pick forward → one look on pre-registered
  criteria → change one thing → re-version → repeat. Never tune on a partial
  sample. Never look twice. CLV outranks W-L at any sample under thousands.

## How to work with Juan
Targeted, justified changes with the reasoning in a comment at the site.
Validate methodology before touching parameters. Verdicts bind evidence,
not product — what the app does with a finding is his decision, recorded in
code with the evidence beside it.

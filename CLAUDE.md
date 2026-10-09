# MLB Prediction Model — working agreement for Claude Code

## What this is
Streamlit app (`mlb_app.py`) deployed on Streamlit Cloud from `main`. The
pick log and daily snapshots live on the `data` branch (`pick_log.json`,
`snapshots/`), written by the app itself via the GitHub Contents API using
secrets configured in Streamlit Cloud. Never commit tokens. Never commit
the log to `main` (commits to `main` reboot the app).

## Current state (2026-09-24)
- `MODEL_VERSION = v4.1-picker-2026-09-24`.
- The 2026 season verdict is in `season_verdict_2026.md`. It is BINDING.
  Do not re-litigate it on 2026 data, and do not fit new parameters to it.
- `ANCHOR_LAMBDA = 1.0` by OWNER DECISION (2026-09-24), not by evidence:
  the pre-registered rule returned 0.00. Juan's product spec is model
  probability → market price → bet/no-bet, and that is the app's job.
  v4.1's picker runs a CHANGED model (double-count fixed, form 15%, tiers
  merged) and is unmeasured; its GOOD-pick record accrues in the tracker.
  The v3 record (49–60, −9.1%/bet) is displayed beside every call. Do not
  hide it, and do not re-argue λ unless Juan raises it.
- `GATE_THRESH_PP`, `ANCHOR_LAMBDA`, `PLAYOFF_BUDGET` are deliberately
  code-only — no UI controls — so in-the-moment eagerness can't loosen
  them. Any commit that changes one must cite the evidence in its message.

## Rules that exist because of real bugs
- All dates go through `now_et()` / `today_et()` (Eastern). Never call
  `datetime.today()` / `datetime.now()` — Streamlit Cloud runs on UTC, and
  naive dates once made every evening session log tomorrow's slate.
- Game identity is `matchup_key(game)`, which suffixes doubleheaders with
  "(G2)". Never key a widget, dict, or log row on "away @ home" alone —
  a doubleheader once took down the whole page.
- Every model change gets a new `MODEL_VERSION` so datasets never mix.
- Every deploy so far has broken something once. Before pushing:
  `python -m py_compile mlb_app.py`, then a local `streamlit run`, then
  check the tracker still loads from the data branch.

## Analysis tooling
- `python log_integrity_check.py pick_log.json` — outcome-free hygiene.
  Run before any analysis; it prints no win rates by design.
- `python season_analysis.py pick_log.json` — the pre-registered workup.
  Criteria are LOCKED. To analyze a new season, copy it, set a new lock
  date in the header before that season's data is complete, then never
  edit the criteria again.
- The method: freeze the model → log every pick forward → one look, on
  pre-registered criteria, at the end → change one thing → re-version →
  repeat. Never tune on a partial sample. Never look twice.

## Evidence-gated change queue
1. Form diagnostic (`form_delta`) ships in v4 → adjudicate Q4 in 2027.
2. Under-dispersion (calibration slope ≈ 2× too flat) — candidate for a
   v4.1 with FRESH data; explicitly not fit on 2026.
3. CLV: log `closing` prices during the playoffs and 2027; the
   beat-the-close rate is the only path to λ > 0.
4. Recent-form window length/weight — revisit only after Q4 has data.

## How to work with Juan
Targeted, justified changes with the reasoning in a code comment at the
site. Validate methodology before touching parameters. When he asks to
loosen a constraint mid-slump, remind him he asked for it to be hard to
loosen — then do what he decides, on the record.

## NBA app (2026-09-27)
- `nba_app.py` is the fourth sibling; its design contract is `NBA_APP_SPEC.md`
  (CLAUDE.md wins on process, the spec wins on design). Pre-registered look:
  `nba_evaluation_plan.md`. Model `nba-v1.0` is FROZEN — no parameter changes
  before that look.
- Its data lives beside the MLB data on the `data` branch: `nba_pick_log.json`
  and `odds_closes/nba/<ET date>.json` (written by
  `scripts/snapshot_closes_nba.py` via `odds_close_snapshot_nba.yml`, which
  only spends a credit when a game tips within 35 min). Never touch
  `mlb_app.py`, `scripts/snapshot_closes.py`, or `odds_close_snapshot.yml`
  for NBA work.
- Game identity is the ESPN event ID; every team join is by ESPN team ID via
  `nba_teams.py`. Widget keys are namespaced `nba1_`.
- Before pushing: `python -m py_compile nba_app.py`, then
  `python -m pytest tests/test_nba_app.py` (AppTest harness on a mock ESPN +
  Odds feed — the only way to exercise the slate without spending credits).
- GitHub's `schedule` trigger fires this repo's crons only 2–4 times a day
  (measured Sep 27 – Oct 9, 2026; MLB too). The real 30-minute cadence comes
  from an external scheduler hitting `workflow_dispatch` — setup and the
  evidence in `nba_closes_scheduler.md`. A "close" quoted hours before tip
  is not a close; check `quoted_at` vs tip before trusting any CLV number.
- Open items Juan supplies before Oct 20: the 30 win totals in
  `nba_preseason_totals_2026.json`, `STAKE_UNITS` / `SEASON_BUDGET` in code,
  the app's own Streamlit Secrets, and the consensus-book list.

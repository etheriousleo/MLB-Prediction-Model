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
  evidence in `nba_closes_scheduler.md`. A dispatch on `ref: main` runs
  main's yml and script, so NBA cron changes only take effect once merged.
  A "close" quoted hours before tip is not a close; check `quoted_at` vs
  tip before trusting any CLV number. The MLB cron spends a credit on every
  run (no window gate) — never put it on a 30-minute dispatcher as is.
- Open items Juan supplies before Oct 20: the 30 win totals in
  `nba_preseason_totals_2026.json`, `STAKE_UNITS` / `SEASON_BUDGET` in code,
  the app's own Streamlit Secrets, and the consensus-book list.

## NBA player production study (2026-10-13)
- Research only: `research/nba_production_study/`, design contract
  `NBA_STUDY_SPEC.md`. No odds, no app, no imports from the app files. The
  step-1 scaffold reached `main` via PR #6 (2026-10-15). Everything after it,
  including the raw pulls, stays on the study branch and is never merged to
  `main`.
- stats.nba.com returns 403 from the Claude Code sandbox, so `pull.py` runs on
  Juan's machine. The parquet pulls are committed, so everything after
  step 1 runs anywhere.
- 2025-26 is the confirm season. Print nothing for it but row counts and
  pass/fail until `LOCKED_HYPOTHESES.md` is committed. Honor the spec's STOP
  points (§11).
- Before pushing: `cd research/nba_production_study && python -m pytest`.
- The lock (`LOCKED_HYPOTHESES.md`, committed) freezes `config.py`, `pull.py`,
  `features.py` and `analyze.py`: `analyze.py --phase confirm` refuses to run
  if any of them differs from the lock commit, and refuses if
  `reports/confirm_2025-26.md` exists. Never edit those files, or delete that
  report, to get a second look.
- **v1 is complete (2026-10-17) and its verdict is BINDING:** 9 of 10 locked
  hypotheses confirmed on 2025-26 (`reports/confirm_2025-26.md`). H6
  (`last5_dev_REB`) is not confirmed: it is significant, but at +0.048 SD it
  is below the 0.05 floor. Don't re-argue the floor or re-run on 2025-26. The
  next step, if any, is the spec's v2: a forward check on 2026-27,
  pre-registered before that season's data accumulates (§12).
- **v2 (set up 2026-10-17):** the 9 confirmed hypotheses, re-tested on
  2026-27 with v1's model, rule, and code. The lock is
  `LOCKED_HYPOTHESES_V2.md` (until Juan approves it, it's
  `LOCKED_HYPOTHESES_V2_DRAFT.md`). From the v2 lock commit until the v2 look,
  `config.py`, `pull.py`, `features.py` and `analyze.py` are frozen:
  `--phase forward` refuses to run if any of them differs. `pull.py` refuses
  to fetch 2026-27 before 2027-04-13. 2026-27's neutral-site games go into
  `data/neutral_site_games.json` (data, not code) after the pull. Then
  `analyze.py --phase forward` runs once. The README has the April steps.

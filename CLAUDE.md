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

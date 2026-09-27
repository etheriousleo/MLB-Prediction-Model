# NBA v1.0 — Pre-registered evaluation plan

**Model:** `nba-v1.0` (frozen 2026-09-27; tag `nba-v1.0-frozen-<date>` before opening night).
**Recorded here and in the `MODEL FREEZE` comment block in `nba_app.py`.** Same method as the
MLB suite: freeze the model → log every pick forward → one look, on pre-registered criteria,
at the end → change one thing → re-version → repeat. Never tune on a partial sample. Never
look twice.

## What is being evaluated

Two paper buckets, evaluated **separately**:

| bucket | pick | logged fields |
|---|---|---|
| `paper_ats` | the model's side with P(cover) > 0.5 at FanDuel's line, cushion ≥ 3pp vs FanDuel's break-even | `verdict_paper_ats`, `paper_ats_side/line/price`, `result_paper_ats`, `clv_paper_ats_pts` |
| `paper_ml` | the model's side with P(win) > 0.5, cushion ≥ 3pp vs FanDuel's break-even | `verdict_paper_ml`, `paper_ml_side/price`, `result_paper_ml`, `clv_paper_ml_pp` |

A pick **counts** only when `paper_counts = True`: both teams fully off the preseason prior
(`w_prior = 0`, i.e. ≥ 15 regular-season games each). The flag is stored at log time and never
recomputed.

Neither paper bucket is ever bet in v1.0. **No bet button is wired to them.** The market gate
(FanDuel vs Pinnacle/consensus, ≥ 1.5pp) is the only thing that produces a bet, from opening
night, and it keeps running in every outcome below.

## When to look

The **earlier** of:

- **200 counted GOOD picks** in a bucket (each bucket looks when it reaches 200), or
- **February 19, 2027** (the All-Star break).

Look **once**. The tracker shows progress (`toward look` column) and never a verdict.

## Pass criteria (per bucket)

1. **Mean CLV** ≥ **+0.5 spread points** (`paper_ats`) or ≥ **+1.0 probability points**
   (`paper_ml`), with the **95% interval excluding zero** (mean ± 1.96 × SE, SE = SD/√n,
   computed from the logged per-pick CLV against the FanDuel close captured by the cron).
2. **Record does not contradict:** ATS hit rate ≥ 48%; ML units at the logged FanDuel prices
   not below −5% per bet. (Win rate over 200 picks has SE ≈ 3.5pp — it cannot acquit or
   convict the model. CLV is the verdict; the record is a sanity check.)

Both must hold. Anything else is a fail.

## Consequences

- **Pass:** that bucket's model gate is enabled as a **second** bet source **beside** the market
  gate, at **0.5 × `STAKE_UNITS`**, for the post-break regular season (~6 weeks, ~400 games).
  Re-version to `nba-v1.1`. Nothing else changes.
- **Fail:** stays paper; nothing changes.
- **Never:** the model gate alone. The market gate keeps running in all cases.

## What may not happen before the look

- No change to `HFA`, `SIGMA`, `FADE_GAMES`, `FORM_WINDOW`, `FORM_WEIGHT`, `B2B_PENALTY`,
  `PRIOR_REGRESS`, the win-total → margin map, `MARKET_EDGE_PP`, or `PAPER_CUSHION_PP`.
- No injury, 3-in-4, travel, or totals inputs (spec §9). These are candidates for the
  **tuning window after** the look, with fresh data — never fit on this season's log.
- No second look, no bucket redefinition, no re-cut by tier or date.

## Data hygiene before the look

- Only rows with `model_version = nba-v1.0` and `paper_counts = True`.
- Only graded rows; pushes count for CLV, are excluded from the record.
- The NBA Cup final (`cup_final = True`) is a real game for the buckets; it is excluded from
  team-strength stats only.
- Postponed rows (`postponed = True`, ungraded) are excluded until they grade under the same
  ESPN event ID.

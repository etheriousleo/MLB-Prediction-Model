# Locked hypotheses: nba-study-v1

> **LOCKED 2026-10-17.** Juan read the draft and approved it in the Claude
> Code session; Claude committed it on his instruction. This commit is the lock
> (spec §7). Nothing below changes, and `config.py`, `pull.py`, `features.py`
> and `analyze.py` are frozen at this commit for Phase B.

Drafted 2026-10-17 from the Phase A report, `reports/explore_2024-25.md`
(code commit `a450781`).

## Decisions behind this list (Juan, 2026-10-17)

- **Selection:** lock every Phase A driver that passed Benjamini-Hochberg at
  10% FDR **and** cleared the practical floor. That gives 10 hypotheses.
- **Practical floor:** 0.05 SD in z, confirmed.
- **Left out:**
  - The arena test. It was significant for AST over the full season
    (p = 0.0014), but the arena effects didn't repeat between Oct–Dec and
    Jan–Apr (r = 0.06), and a 29-parameter F-test has no single signed
    coefficient to lock.
  - Three drivers that passed BH but fall below the floor:
    `opp_oreb_pct` on REB (−0.040), `opp_3pa_rate` on REB (+0.028), and
    `opp_pace` on AST (+0.036).
- **Model departures from the spec, approved:**
  - Each stat's model carries its own `miss_S_share`, not `miss_usage_share`
    as well. The two correlate at r = 0.91–0.995.
  - `min_resid` is fit per stat. It isn't part of Phase B.
  - Neutral-site games are never analysis rows (step 1).

## The hypotheses

All 10 predict a game **above** the player's own norm (positive direction).

| ID | Stat | Feature | Direction | 2024-25 effect on z [95% CI] | 2024-25 hit-rate pts | 2024-25 q (BH) | Channel |
|---|---|---|---|---|---|---|---|
| H1 | PTS | `miss_PTS_share` | + | +0.152 [+0.127, +0.177] | +5.43 | 3.5e-30 | both |
| H2 | PTS | `last5_min_dev` | + | +0.064 [+0.040, +0.089] | +2.77 | 2.1e-06 | offsetting |
| H3 | PTS | `last5_dev_PTS` | + | +0.058 [+0.030, +0.086] | +2.26 | 0.00031 | rate |
| H4 | REB | `miss_REB_share` | + | +0.084 [+0.063, +0.105] | +3.16 | 1.2e-13 | both |
| H5 | REB | `last5_min_dev` | + | +0.092 [+0.069, +0.115] | +3.48 | 1.2e-13 | minutes |
| H6 | REB | `last5_dev_REB` | + | +0.084 [+0.059, +0.109] | +2.97 | 6.4e-10 | both |
| H7 | AST | `miss_AST_share` | + | +0.119 [+0.091, +0.146] | +3.61 | 1.3e-15 | both |
| H8 | AST | `last5_dev_AST` | + | +0.087 [+0.061, +0.113] | +3.93 | 6.2e-10 | rate |
| H9 | AST | `last5_min_dev` | + | +0.077 [+0.049, +0.104] | +2.56 | 2.3e-07 | minutes |
| H10 | AST | `opp_ast_allowed` | + | +0.067 [+0.025, +0.110] | +2.48 | 0.0091 | rate |

In plain terms:

- **H1, H4, H7 (fresh teammate absences):** when regulars who were playing
  recently sit tonight, the remaining players' points, rebounds, and assists
  rise above their norms. The effect is per 1 SD of the missing share of the
  team's usual output in that stat.
- **H3, H6, H8 (recent form):** a player above his stint average over his last
  5 games is above it again tonight. That is persistence, not mean reversion.
- **H2, H5, H9 (recent minutes):** a player whose minutes over the last 5
  games run above his stint average produces more tonight, mostly through
  minutes.
- **H10 (opponent):** opponents who allow more assists per 100 possessions
  give up more assists, through the per-minute rate.

## Metric and model (exactly as run in Phase A)

- **Metric:** the joint-model OLS coefficient on `z_S` for the hypothesis's
  feature. `z_S` is (game stat − stint baseline) / stint SD, using prior games
  in the stint only.
- **Sample:** the season's player-games with `elig_S`. That means at least 10
  prior games in the stint, a pre-game average of at least 20 minutes, not a
  neutral-site game, and a baseline at or above the stat floor (PTS 8, REB 3,
  AST 2).
- **Regressors in each stat's model:** `is_home`, `days_rest`, `b2b`, `opp_b2b`, `three_in_four`, `opp_pace`, `opp_drtg`, `opp_efg_allowed`, `opp_ast_allowed`, `opp_fg_pct`, `opp_3pa_rate`, `opp_oreb_pct`, `opp_dreb_pct`, `net_gap`, `abs_net_gap`, `last5_min_dev`, plus `miss_S_share` and
  `last5_dev_S` for that stat, `season_phase` dummies (the earliest month
  bucket present is the reference), and a constant. Continuous drivers are
  standardized within that sample; the flags (`is_home`, `b2b`, `opp_b2b`,
  `three_in_four`) stay 0/1.
- **Inference:** SEs clustered by player, Wald z, two-sided 95% CIs.
- **Code:** `config.py`, `pull.py`, `features.py`, and `analyze.py` as of the
  lock commit. Phase B refuses to run if any of them differs from that commit.

## Confirmation rule (fixed in the spec before Phase A, §7)

A hypothesis is **confirmed** on 2025-26 only if all three hold:

1. Its coefficient has the same sign as in Phase A (the locked direction).
2. Its one-sided p in the locked direction is below 0.05 after Holm
   correction across all 10 hypotheses.
3. Its point estimate is at least 0.05 SD in the locked direction.

## What Phase B does (§8)

`python analyze.py --phase confirm` refuses to run unless this file is
committed as `LOCKED_HYPOTHESES.md` with status `locked`, has no uncommitted
changes, and the four code files are unchanged since its commit. It also
refuses if `reports/confirm_2025-26.md` already exists, so there is only one
look. Only after those checks does it touch 2025-26, and then it runs only the
three locked joint models. It records this file's commit hash, and it reports
each hypothesis as confirmed or not, with the 2025-26 effect and CI beside the
2024-25 one.

## Machine-readable block (read by `analyze.py --phase confirm`)

```json
{
  "study_version": "nba-study-v1",
  "status": "locked",
  "approved_by": "Juan",
  "approved_on": "2026-10-17",
  "drafted_on": "2026-10-17",
  "explore_report": "reports/explore_2024-25.md",
  "explore_commit": "a450781",
  "practical_floor_sd": 0.05,
  "alpha_one_sided": 0.05,
  "correction": "holm",
  "model": {
    "outcome": "z_{s}",
    "sample": "elig_{s}",
    "estimator": "OLS",
    "se": "clustered by PLAYER_ID",
    "common_drivers": [
      "is_home",
      "days_rest",
      "b2b",
      "opp_b2b",
      "three_in_four",
      "opp_pace",
      "opp_drtg",
      "opp_efg_allowed",
      "opp_ast_allowed",
      "opp_fg_pct",
      "opp_3pa_rate",
      "opp_oreb_pct",
      "opp_dreb_pct",
      "net_gap",
      "abs_net_gap",
      "last5_min_dev"
    ],
    "per_stat_drivers": [
      "miss_{s}_share",
      "last5_dev_{s}"
    ],
    "binary_drivers": [
      "is_home",
      "b2b",
      "opp_b2b",
      "three_in_four"
    ],
    "control": "season_phase dummies; the earliest month bucket present is the reference",
    "standardization": "continuous drivers z-scored within the season's elig_{s} rows; flags 0/1"
  },
  "hypotheses": [
    {
      "id": "H1",
      "stat": "PTS",
      "feature": "miss_PTS_share",
      "direction": "+",
      "explore_coef": 0.151859,
      "explore_ci": [
        0.126547,
        0.177172
      ],
      "explore_p": 6.39e-32,
      "explore_q": 3.45e-30,
      "explore_hit_pts": 5.43,
      "channel": "both"
    },
    {
      "id": "H2",
      "stat": "PTS",
      "feature": "last5_min_dev",
      "direction": "+",
      "explore_coef": 0.064488,
      "explore_ci": [
        0.039766,
        0.08921
      ],
      "explore_p": 3.18e-07,
      "explore_q": 2.14e-06,
      "explore_hit_pts": 2.77,
      "channel": "offsetting"
    },
    {
      "id": "H3",
      "stat": "PTS",
      "feature": "last5_dev_PTS",
      "direction": "+",
      "explore_coef": 0.057718,
      "explore_ci": [
        0.029793,
        0.085642
      ],
      "explore_p": 5.1e-05,
      "explore_q": 0.000306,
      "explore_hit_pts": 2.26,
      "channel": "rate"
    },
    {
      "id": "H4",
      "stat": "REB",
      "feature": "miss_REB_share",
      "direction": "+",
      "explore_coef": 0.083528,
      "explore_ci": [
        0.062503,
        0.104553
      ],
      "explore_p": 6.88e-15,
      "explore_q": 1.24e-13,
      "explore_hit_pts": 3.16,
      "channel": "both"
    },
    {
      "id": "H5",
      "stat": "REB",
      "feature": "last5_min_dev",
      "direction": "+",
      "explore_coef": 0.091925,
      "explore_ci": [
        0.068678,
        0.115173
      ],
      "explore_p": 9.18e-15,
      "explore_q": 1.24e-13,
      "explore_hit_pts": 3.48,
      "channel": "minutes"
    },
    {
      "id": "H6",
      "stat": "REB",
      "feature": "last5_dev_REB",
      "direction": "+",
      "explore_coef": 0.083656,
      "explore_ci": [
        0.058502,
        0.10881
      ],
      "explore_p": 7.1e-11,
      "explore_q": 6.39e-10,
      "explore_hit_pts": 2.97,
      "channel": "both"
    },
    {
      "id": "H7",
      "stat": "AST",
      "feature": "miss_AST_share",
      "direction": "+",
      "explore_coef": 0.118581,
      "explore_ci": [
        0.090896,
        0.146265
      ],
      "explore_p": 4.65e-17,
      "explore_q": 1.26e-15,
      "explore_hit_pts": 3.61,
      "channel": "both"
    },
    {
      "id": "H8",
      "stat": "AST",
      "feature": "last5_dev_AST",
      "direction": "+",
      "explore_coef": 0.086893,
      "explore_ci": [
        0.060895,
        0.11289
      ],
      "explore_p": 5.72e-11,
      "explore_q": 6.18e-10,
      "explore_hit_pts": 3.93,
      "channel": "rate"
    },
    {
      "id": "H9",
      "stat": "AST",
      "feature": "last5_min_dev",
      "direction": "+",
      "explore_coef": 0.076542,
      "explore_ci": [
        0.049458,
        0.103627
      ],
      "explore_p": 3.04e-08,
      "explore_q": 2.35e-07,
      "explore_hit_pts": 2.56,
      "channel": "minutes"
    },
    {
      "id": "H10",
      "stat": "AST",
      "feature": "opp_ast_allowed",
      "direction": "+",
      "explore_coef": 0.067233,
      "explore_ci": [
        0.024921,
        0.109544
      ],
      "explore_p": 0.00184,
      "explore_q": 0.00905,
      "explore_hit_pts": 2.48,
      "channel": "rate"
    }
  ]
}
```

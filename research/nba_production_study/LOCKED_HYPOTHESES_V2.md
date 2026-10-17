# Locked hypotheses, v2: forward check on 2026-27

> **LOCKED 2026-10-17, before opening night (2026-10-20).** Juan read the
> draft and approved it in the Claude Code session; Claude committed it on his
> instruction. This commit is the v2 lock. Nothing below changes, and
> `config.py`, `pull.py`, `features.py` and `analyze.py` are frozen at this
> commit until the v2 look.

Drafted 2026-10-17. v2 is the spec's forward check (§12): v1's confirmed
hypotheses, tested again on a season that hadn't started when they were
locked. Nothing is re-fit, re-selected, or re-tuned. The model, the rule, the
floor, and the code are v1's.

## Where this comes from

- **v1 lock:** `LOCKED_HYPOTHESES.md` at commit `19573ee`, 10 hypotheses.
- **v1 result:** `reports/confirm_2025-26.md` at `bd54f32`. 9 of 10 were
  confirmed on 2025-26.
- **Dropped:** H6 (`last5_dev_REB`) was not confirmed. It was significant, but
  at +0.048 SD it fell below the 0.05 floor, and that verdict is binding. The
  IDs stay as they were in v1, so the gap is visible.

## The hypotheses

All 9 predict a game **above** the player's own norm.

| ID | Stat | Feature | Direction | 2024-25 effect on z [95% CI] | 2025-26 effect on z [95% CI] |
|---|---|---|---|---|---|
| H1 | PTS | `miss_PTS_share` | + | +0.152 [+0.127, +0.177] | +0.153 [+0.129, +0.176] |
| H2 | PTS | `last5_min_dev` | + | +0.064 [+0.040, +0.089] | +0.084 [+0.058, +0.109] |
| H3 | PTS | `last5_dev_PTS` | + | +0.058 [+0.030, +0.086] | +0.070 [+0.043, +0.097] |
| H4 | REB | `miss_REB_share` | + | +0.084 [+0.063, +0.105] | +0.077 [+0.051, +0.103] |
| H5 | REB | `last5_min_dev` | + | +0.092 [+0.069, +0.115] | +0.089 [+0.064, +0.115] |
| H7 | AST | `miss_AST_share` | + | +0.119 [+0.091, +0.146] | +0.099 [+0.070, +0.129] |
| H8 | AST | `last5_dev_AST` | + | +0.087 [+0.061, +0.113] | +0.084 [+0.058, +0.111] |
| H9 | AST | `last5_min_dev` | + | +0.077 [+0.049, +0.104] | +0.100 [+0.076, +0.125] |
| H10 | AST | `opp_ast_allowed` | + | +0.067 [+0.025, +0.110] | +0.065 [+0.017, +0.114] |

## Metric, model, and rule (unchanged from v1)

- **Metric:** the joint-model OLS coefficient on `z_S` for the hypothesis's
  feature.
- **Sample:** 2026-27 regular-season player-games with `elig_S`. That means
  at least 10 prior games in the stint, a pre-game average of at least 20
  minutes, not a neutral-site game, and a baseline at or above the stat floor
  (PTS 8, REB 3, AST 2).
- **Regressors in each stat's model:** `is_home`, `days_rest`, `b2b`, `opp_b2b`, `three_in_four`, `opp_pace`, `opp_drtg`, `opp_efg_allowed`, `opp_ast_allowed`, `opp_fg_pct`, `opp_3pa_rate`, `opp_oreb_pct`, `opp_dreb_pct`, `net_gap`, `abs_net_gap`, `last5_min_dev`, plus `miss_S_share` and
  `last5_dev_S` for that stat, `season_phase` dummies, and a constant.
  Continuous drivers are standardized within that sample; flags stay 0/1.
  SEs are clustered by player.
- **Confirmed on 2026-27** only if all three hold:
  1. Its coefficient has the locked sign.
  2. Its one-sided p is below 0.05 after Holm correction across these 9.
  3. Its point estimate is at least 0.05 SD in the locked direction.

## Timeline and safeguards

1. **Now (before 2026-10-20):** this lock is committed. From that commit,
   `config.py`, `pull.py`, `features.py`, and `analyze.py` are frozen until
   the v2 look; `--phase forward` refuses to run if any of them differs.
2. **During the season:** nothing. `pull.py` refuses to fetch 2026-27 before
   **2027-04-13** (the regular season ends 2027-04-11),
   so there is no mid-season peek.
3. **After 2027-04-13:** Juan runs `pull.py` locally and
   pushes the raw files. The sanity check will list 2026-27's neutral-site
   games (both rows "@"). They go into `data/neutral_site_games.json` after
   being checked against the published schedule. That's a data file, not
   frozen code, and the check accepts only the exact set the data shows.
4. **Then, once:** `python analyze.py --phase forward` writes
   `reports/forward_2026-27.md` and never runs again.

**If something forces a code change** (the season ends short of 1,230
games, or the data format shifts), stop before any look. Document the reason,
re-lock, and only then run.

## Machine-readable block (read by `analyze.py --phase forward`)

```json
{
  "study_version": "nba-study-v2",
  "status": "locked",
  "approved_by": "Juan",
  "approved_on": "2026-10-17",
  "drafted_on": "2026-10-17",
  "season": "2026-27",
  "basis": {
    "v1_lock": "LOCKED_HYPOTHESES.md at 19573ee",
    "v1_result": "reports/confirm_2025-26.md at bd54f32: 9 of 10 confirmed"
  },
  "pull_not_before": "2027-04-13",
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
      "confirm_coef": 0.153,
      "confirm_ci": [
        0.129,
        0.176
      ]
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
      "confirm_coef": 0.084,
      "confirm_ci": [
        0.058,
        0.109
      ]
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
      "confirm_coef": 0.07,
      "confirm_ci": [
        0.043,
        0.097
      ]
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
      "confirm_coef": 0.077,
      "confirm_ci": [
        0.051,
        0.103
      ]
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
      "confirm_coef": 0.089,
      "confirm_ci": [
        0.064,
        0.115
      ]
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
      "confirm_coef": 0.099,
      "confirm_ci": [
        0.07,
        0.129
      ]
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
      "confirm_coef": 0.084,
      "confirm_ci": [
        0.058,
        0.111
      ]
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
      "confirm_coef": 0.1,
      "confirm_ci": [
        0.076,
        0.125
      ]
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
      "confirm_coef": 0.065,
      "confirm_ci": [
        0.017,
        0.114
      ]
    }
  ]
}
```

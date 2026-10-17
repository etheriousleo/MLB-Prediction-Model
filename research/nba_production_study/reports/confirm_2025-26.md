# Phase B: confirm 2025-26

`nba-study-v1`, generated 2026-10-17 00:39 UTC by `analyze.py --phase confirm` at commit `19573ee`. Lock: `LOCKED_HYPOTHESES.md` at commit `19573ee07b02512b2a208d01a3ae6ad1ec4191cc`, approved by Juan on 2026-10-17. Raw data: `2025-26_player.parquet` sha256 `eec2c5c566f4…`, `2025-26_team.parquet` sha256 `c8e4db7cc976…`. All tests passed before this report was written.

This is the study's one look at 2025-26 (§8). Only the locked models ran: no new features, buckets, or exploratory tables.

## Result: 9 of 10 hypotheses confirmed

A hypothesis is confirmed only if all three hold (§7, fixed before Phase A): the 2025-26 effect has the locked sign; its one-sided p in the locked direction is below 0.05 after Holm correction across all 10 locked hypotheses; and the point estimate is at least 0.05 SD in the locked direction.

| ID | Stat | Feature | Direction | 2024-25 effect on z [95% CI] | 2025-26 effect on z [95% CI] | One-sided p | Holm p | ≥ floor | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1 | PTS | `miss_PTS_share` | + | +0.152 [+0.127, +0.177] | +0.153 [+0.129, +0.176] | 1.9e-37 | 1.9e-36 | yes | **confirmed** |
| H2 | PTS | `last5_min_dev` | + | +0.064 [+0.040, +0.089] | +0.084 [+0.058, +0.109] | 7.5e-11 | 4.5e-10 | yes | **confirmed** |
| H3 | PTS | `last5_dev_PTS` | + | +0.058 [+0.030, +0.086] | +0.070 [+0.043, +0.097] | 1.5e-07 | 4.6e-07 | yes | **confirmed** |
| H4 | REB | `miss_REB_share` | + | +0.084 [+0.063, +0.105] | +0.077 [+0.051, +0.103] | 2.6e-09 | 1e-08 | yes | **confirmed** |
| H5 | REB | `last5_min_dev` | + | +0.092 [+0.069, +0.115] | +0.089 [+0.064, +0.115] | 5e-12 | 4e-11 | yes | **confirmed** |
| H6 | REB | `last5_dev_REB` | + | +0.084 [+0.059, +0.109] | +0.048 [+0.024, +0.072] | 3.8e-05 | 7.6e-05 | no | not confirmed |
| H7 | AST | `miss_AST_share` | + | +0.119 [+0.091, +0.146] | +0.099 [+0.070, +0.129] | 1.8e-11 | 1.3e-10 | yes | **confirmed** |
| H8 | AST | `last5_dev_AST` | + | +0.087 [+0.061, +0.113] | +0.084 [+0.058, +0.111] | 2.2e-10 | 1.1e-09 | yes | **confirmed** |
| H9 | AST | `last5_min_dev` | + | +0.077 [+0.049, +0.104] | +0.100 [+0.076, +0.125] | 4.9e-16 | 4.4e-15 | yes | **confirmed** |
| H10 | AST | `opp_ast_allowed` | + | +0.067 [+0.025, +0.110] | +0.065 [+0.017, +0.114] | 0.0043 | 0.0043 | yes | **confirmed** |

## Samples

| Stat | Player-games | Players |
|---|---|---|
| PTS | 11,398 | 285 |
| REB | 10,467 | 269 |
| AST | 8,682 | 230 |

## Locked model

Each stat's joint model, exactly as locked: OLS of `z_S` on `is_home`, `days_rest`, `b2b`, `opp_b2b`, `three_in_four`, `opp_pace`, `opp_drtg`, `opp_efg_allowed`, `opp_ast_allowed`, `opp_fg_pct`, `opp_3pa_rate`, `opp_oreb_pct`, `opp_dreb_pct`, `net_gap`, `abs_net_gap`, `last5_min_dev`, `miss_S_share`, `last5_dev_S`, season_phase dummies and a constant, on the season's `elig_S` rows. Continuous drivers are standardized within that sample; flags stay 0/1. SEs are clustered by player. Only the locked coefficients are reported.


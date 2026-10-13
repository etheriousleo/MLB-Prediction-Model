# NBA Player Production Study: Build Spec (`nba-study-v1`)

For Claude Code. Place at the repo root next to `CLAUDE.md` and `NBA_APP_SPEC.md`. Read `CLAUDE.md` first; its discipline rules apply here too.

Drafted 2026-10-12.

---

## 1. The question

For rotation players, which **pre-game** factors predict a game in which a player's **points, rebounds, or assists** come in above his own norm?

This is analytics only. There are no props, odds, betting, or market data anywhere in this study. It is not an app: it is scripts, tests, and markdown reports.

---

## 2. Ground rules

- **Branch.** Work on a `nba-study` branch. Do not merge it to `main`, because commits to `main` reboot the live Streamlit apps. Do not modify or import anything from the app files.
- **Run locally.** stats.nba.com blocks most cloud IPs, so the pull must run on Juan's machine.
- **Separate dependencies.** Use a `requirements.txt` inside the study folder. Do not touch the root `requirements.txt`, because Streamlit Cloud installs from it.
- **No look-ahead.** Every feature for a game uses only games played on strictly earlier dates. This is enforced by tests (section 9), not by convention.
- **Pre-register, look once.** 2024-25 is the exploration season. 2025-26 is touched once, only after Juan approves the locked hypothesis list (section 7).
- **IDs, not names.** Join on `PLAYER_ID`, `TEAM_ID`, and `GAME_ID`.
- **Constants live in `config.py`.** That includes seasons, thresholds, floors, and the shrinkage constant. Nothing is tunable from the command line.
- Write clean, commented code.

---

## 3. Seasons

| Role | Season | nba_api string |
|---|---|---|
| Explore | 2024-25 | `"2024-25"` |
| Confirm | 2025-26 | `"2025-26"` |

Regular season only. Change these only before any analysis has run.

---

## 4. Data pull (`pull.py`)

Make two calls per season with `nba_api.stats.endpoints.leaguegamelog.LeagueGameLog`, using `season_type_all_star="Regular Season"`:

- `player_or_team_abbreviation="P"` returns player box scores.
- `player_or_team_abbreviation="T"` returns team box scores.

Call settings: `timeout=60`, 3 retries with backoff, and a 2-second pause between calls.

Save raw output to `data/raw/{season}_{player|team}.parquet` and commit those files, so results reproduce without hitting the API again. Skip a call if its file already exists, unless `--refresh` is passed.

**Sanity assertions** (fail loudly if any of these is false):

- Each season has exactly 1,230 unique `GAME_ID`s. Each game has exactly two team rows: one home (`"vs."` in `MATCHUP`) and one away (`"@"`).
- For every team-game, the sum of player `PTS` equals the team `PTS`.
- The NBA Cup championship game does not appear, since it doesn't count toward regular-season stats. If the unique game count isn't 1,230, stop and report the discrepancy.
- Check whether players who did not play appear in the player log (with `MIN` of 0 or null). Teammate-absence detection (section 5.3) depends on the answer, so document what you find.

For 2025-26, print row counts only. Do not produce any outcome summaries for that season before the lock.

---

## 5. Definitions (`features.py`)

### 5.1 Unit and eligibility

- **Unit:** one player-game with `MIN > 0`.
- **Stints:** baselines are computed per player-team stint and reset when a player is traded.
- **Eligibility:** a player-game is eligible when the player has at least 10 prior games in the stint and a pre-game average of at least 20 minutes.
- **Stat-specific floors:** a player-game is dropped from one stat's analysis (and only that stat's) if his baseline for that stat is below the floor. The floors are PTS ≥ 8, REB ≥ 3, AST ≥ 2. This keeps whole-number effects on low counts from dominating the results.

### 5.2 Baseline and outcomes

These apply to each stat S in {PTS, REB, AST}. All baselines use prior games in the stint only; the current game is excluded.

| Name | Definition |
|---|---|
| `base_S` | Mean of S over prior games in the stint |
| `sd_S` | Sample SD of S over the same games, floored at 1.0 |
| `base_min` | Mean minutes over prior games |
| `base_rate_S` | sum(S) / sum(MIN) over prior games |
| **`z_S`** | (S − `base_S`) / `sd_S`. **Primary outcome.** |
| `hit_S` | 1 if S > `base_S`. Secondary outcome, reported for readability. |
| `min_resid` | MIN − `base_min`. Shared across all three stats. |
| `rate_resid_S` | S/MIN − `base_rate_S`. Only for games with MIN ≥ 10. |

Counting stats are right-skewed, so the base rate of `hit_S` will sit below 50%. Report the actual base rate and the median z, and judge every factor against the base rate, not against 50%.

### 5.3 Pre-game features

All features use strictly earlier dates only.

Team and opponent rates are season-to-date averages, shrunk toward the league average: `x = (n·team + k·league) / (n + k)` with `k = 5`.

Possessions per game: `POSS = FGA + 0.44·FTA − OREB + TOV`, averaged over the two teams in the game.

**Context**

| Feature | Definition | Hypothesis target |
|---|---|---|
| `is_home` | `MATCHUP` contains `"vs."` | All |
| `days_rest` | Days since the team's previous game, minus 1 | All |
| `b2b` | `days_rest == 0` | Minutes |
| `opp_b2b` | Opponent is on a back-to-back | All |
| `three_in_four` | Third game in four nights | Minutes |
| `season_phase` | Month bucket | Control |

**Opponent**

| Feature | Definition | Hypothesis target |
|---|---|---|
| `opp_pace` | Opponent possessions per 48 minutes | All |
| `opp_drtg` | Points the opponent allows per 100 possessions | PTS |
| `opp_efg_allowed` | Opponent's defensive eFG% | PTS, AST |
| `opp_ast_allowed` | Assists the opponent allows per 100 possessions | AST |
| `opp_fg_pct` | Opponent's own shooting FG%. Their misses become our defensive boards. | REB |
| `opp_3pa_rate` | Opponent's 3PA/FGA. More threes means more long rebounds. | REB |
| `opp_oreb_pct` | Opponent's offensive rebound rate across its games, i.e. competition on our defensive glass | REB |
| `opp_dreb_pct` | Opponent's defensive rebound rate across its games, i.e. room for our offensive boards | REB |

**Teammates (fresh absences only)**

- A **regular** is a teammate with at least 10 games in his stint and a pre-game average of at least 20 minutes.
- A **fresh absence** is a regular who is not in tonight's box score but played in at least one of the team's previous 5 games. Long absences are excluded because they are already baked into the player's baseline.

| Feature | Definition | Hypothesis target |
|---|---|---|
| `miss_usage_share` | Sum of fresh-absent regulars' pre-game average of (FGA + 0.44·FTA + TOV), divided by the team's pre-game average of the same | PTS, AST |
| `miss_S_share` | The same construction, using stat S | Per stat |

Limitation, to be documented in the report: this uses actual absence. A late scratch isn't knowable before the game in real life. That's acceptable for a drivers study.

**Game script**

| Feature | Definition | Hypothesis target |
|---|---|---|
| `net_gap` | Team's pre-game net rating minus the opponent's | Minutes |
| `abs_net_gap` | Absolute value of `net_gap`, as a blowout-risk proxy | Minutes |

**Form**

| Feature | Definition | Hypothesis target |
|---|---|---|
| `last5_dev_S` | (mean of S over last 5 games − `base_S`) / `sd_S` | Tests mean reversion vs. hot hand |
| `last5_min_dev` | Mean minutes over last 5 games − `base_min` | Minutes |

**Arena**

| Feature | Definition | Hypothesis target |
|---|---|---|
| `arena_team_id` | Home team's `TEAM_ID`, as a proxy for the scorekeeper | AST and REB. PTS serves as the control. |

### 5.4 Post-game diagnostics (never used in driver models)

- `ot_game`: team MIN > 240
- Actual final margin
- `early_exit`: MIN < 0.5 · `base_min`

These are used only to explain variance and for sensitivity runs.

---

## 6. Phase A: Explore 2024-25 (`analyze.py --phase explore`)

Load 2024-25 only.

1. **Base rates.** For each stat, report the hit rate, mean and median z, and a plot of the z distribution.
2. **One factor at a time.** Bucket each feature: binary flags as-is, continuous features in quintiles. For each bucket, report n, hit rate, mean z, and a 95% CI clustered by player.
3. **Joint model per stat.** Fit OLS of `z_S` on all features plus `season_phase`, with SEs clustered by player. Fit a logistic model on `hit_S` as a companion. Standardize continuous features so coefficients are comparable.
4. **Minutes vs. rate split.** Fit the same joint model on `min_resid` (once, since it's shared) and on standardized `rate_resid_S` for each stat. For each driver, report whether it acts through minutes, per-minute rate, or both.
5. **Arena test.** Add arena fixed effects to each stat's joint model and F-test their joint significance. As a stability check, correlate arena effects estimated on Oct–Dec games with those estimated on Jan–Apr games.
   - Expected result: real for AST, possibly REB, null for PTS.
   - A strong PTS arena effect would mean something else is wrong.
6. **Sensitivity.** Rerun the joint models excluding OT games, then excluding early exits.
7. **Multiple testing.** Apply Benjamini-Hochberg at a 10% FDR across all Phase A joint-model coefficients, and flag the candidates that pass.

**Report:** write `reports/explore_2024-25.md`, with figures in `reports/figures/`.

- Lead with a ranked table of drivers for each stat.
- For each driver, show the effect in z units and in hit-rate points, its CI, its channel (minutes or rate), and its BH flag.

**STOP after this report.**

---

## 7. The lock (`LOCKED_HYPOTHESES.md`)

Claude Code drafts this file from the Phase A results. Juan edits and approves it, and it is committed before Phase B runs.

Each hypothesis states:

- the stat
- the feature
- the direction
- the metric (the joint-model coefficient on `z_S`)
- the exact model specification as run in Phase A

**Practical floor (proposed):** |effect| ≥ 0.05 SD in z. Juan confirms or changes this at the lock.

**Confirmation rule (fixed now):** a hypothesis is confirmed only if all three hold:

- same sign as in Phase A
- one-sided p < 0.05 in the locked direction, after Holm correction across the locked list
- effect size ≥ the practical floor

---

## 8. Phase B: Confirm 2025-26 (`analyze.py --phase confirm`)

- Refuse to run unless `LOCKED_HYPOTHESES.md` exists and is committed with no uncommitted changes.
- Record the lock file's commit hash in the report.
- Run only the locked models. No new features, buckets, or exploratory tables.
- Write `reports/confirm_2025-26.md`. Refuse to run if that file already exists, so there is only one look.
- For each hypothesis, report confirmed or not confirmed, with the 2025-26 effect and CI shown beside the 2024-25 effect.

---

## 9. Tests (`tests/`, pytest)

All tests must pass before any report is written.

- `test_baseline_excludes_current`: a synthetic player whose baseline is computed by hand.
- `test_future_perturbation`: randomize all stats on dates after a cutoff date D. Every player, opponent, and teammate feature for games dated on or before D must be unchanged.
- `test_stint_reset`: a traded player's baseline restarts with the new team.
- `test_fresh_absence`: a synthetic roster in which one regular sits once and another sits 15 straight games. Only the first counts as a fresh absence.
- `test_pull_sanity`: the section 4 assertions, run against the committed raw data.

---

## 10. Layout

```
NBA_STUDY_SPEC.md                  (this file, repo root)
research/nba_production_study/
  README.md                        how to run, in order
  requirements.txt                 nba_api, pandas, numpy, statsmodels, matplotlib, pyarrow, pytest
  config.py                        all constants
  pull.py
  features.py
  analyze.py
  LOCKED_HYPOTHESES.md             created at the lock
  data/raw/                        committed parquet pulls
  data/features/
  reports/
  tests/
```

---

## 11. Build order and stop points

1. Pull both seasons and run the sanity checks. **STOP:** report row counts and the finding on non-playing players.
2. Build features and get all tests passing.
3. Produce the Phase A report. **STOP:** Juan reviews it.
4. Draft the lock file. Juan approves and commits it.
5. Run Phase B once and write its report.

---

## 12. Out of scope for v1

- Props, odds, and betting of any kind
- A Streamlit page
- Play-by-play, tracking, or injury-report data
- Player positions
- Blocks and steals (dropped by Juan)
- Playoffs

A forward check on the 2026-27 season is a possible v2 if anything survives Phase B.

# NBA player production study (`nba-study-v1`)

The question: for rotation players, which **pre-game** factors predict a game
where a player's points, rebounds, or assists come in above his own norm?
This is analytics only, with no odds or betting. It is scripts, tests, and
markdown reports, not an app.

- **Design contract:** [`NBA_STUDY_SPEC.md`](../../NBA_STUDY_SPEC.md) at the repo root.
- **Process:** [`CLAUDE.md`](../../CLAUDE.md). Freeze, pre-register, and look once.

## Status

| Step | What | State |
|---|---|---|
| 1 | Pull both seasons, sanity checks | **Done** (2026-10-16). Findings: `reports/step1_findings.md`. |
| 2 | `features.py` + leakage tests | **Done** (2026-10-17). All §9 tests pass on synthetic and real 2024-25 data. |
| 3 | Phase A: explore 2024-25 | **Done** (2026-10-17): `reports/explore_2024-25.md`. Reviewed and approved by Juan. |
| 4 | `LOCKED_HYPOTHESES.md` | Draft ready: `LOCKED_HYPOTHESES_DRAFT.md`. **Waiting on Juan's approval.** |
| 5 | Phase B: confirm 2025-26, once | Code ready and tested. Runs right after the lock commit. |

## Ground rules

- **Never merge this branch to `main`.** Commits to `main` reboot the live Streamlit apps.
- **Pull locally.** stats.nba.com blocks most cloud IPs. It returned 403 from the
  Claude Code sandbox, so `pull.py` has to run on your own machine.
- **Separate dependencies.** Use `requirements.txt` in this folder. The root one is
  what Streamlit Cloud installs.
- **No look-ahead.** Features use strictly earlier dates only, and tests enforce
  this (§9).
- **2025-26 stays sealed until the lock.** `pull.py` prints only row counts and
  pass/fail for it, and failure messages name IDs, never stat values.
- **All constants live in `config.py`.** Nothing is tunable from the command line;
  `--refresh` is the only flag.
- **Join on IDs** (`PLAYER_ID`, `TEAM_ID`, `GAME_ID`), never on names.
- **Don't import or modify the app files.**

## Setup (once)

Use Python 3.10 or newer.

```bash
cd research/nba_production_study
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run, in order

### 1. Pull and sanity-check (on your machine)

Work on the study branch, never `main`. The study branch is currently
`claude/fervent-hopper-ziawpi`.

```bash
git fetch origin
git checkout claude/fervent-hopper-ziawpi
git pull origin claude/fervent-hopper-ziawpi
cd research/nba_production_study
python pull.py          # four API calls, a 2 s pause between them, about a minute
python -m pytest        # test_pull_sanity now runs against the real files
```

Each call has a 60 s timeout and 3 retries with 5/10/20 s backoff. A file that
already exists is skipped, so if the API throttles you partway through,
re-running picks up where it left off. `python pull.py --refresh` re-pulls
everything.

`pull.py` asserts the following and exits non-zero if any check fails:

- 1,230 unique games, with 30 teams at 82 games each
- No NBA Cup final (its GAME_ID prefix is `006`)
- One home and one away row per game
- No duplicate rows
- The player and team logs cover the same team-games
- Player PTS sum to team PTS for every team-game

It writes `reports/pull_sanity.md`, which has row counts, each check's result,
and the evidence on whether players who didn't play appear in the player log.

Then commit and push the results:

```bash
git add data/raw reports/pull_sanity.md
git commit -m "nba-study: raw pulls for 2024-25 and 2025-26"
git push origin claude/fervent-hopper-ziawpi
```

Don't open or merge a pull request for the study branch. Every merge to
`main` restarts the live apps, and the raw pulls don't belong there.

**STOP.** Report the row counts and the non-playing-player finding (§11.1).

### 2. Features and tests

```bash
python features.py      # writes data/features/2024-25.parquet (not committed; rebuilt on demand)
python -m pytest
```

`features.py` builds the §5 definitions for every player-game with MIN > 0,
and builds the explore season only: it refuses the confirm season until the
lock. Its docstring lists the choices the spec leaves open. Neutral-site games
count in every backward-looking calculation but are never analysis rows
(Juan, 2026-10-17).

The §9 tests in `tests/test_features.py` include the no-look-ahead check: every
stat after a cutoff date is randomized, and nothing on or before that date may
move. It runs on a synthetic league and on the real 2024-25 logs. A stricter
variant also randomizes the cutoff day's own stats and requires every
pre-game value on that day to hold. All tests must pass before any report is
written.

### 3. Phase A

```bash
python analyze.py --phase explore   # runs the tests first; ~30 s
```

It writes `reports/explore_2024-25.md`, with figures in `reports/figures/`. The
report leads with the lock candidates (drivers that pass Benjamini-Hochberg at
10% FDR and the proposed 0.05 SD floor), then gives a ranked driver table per
stat, base rates, the arena test, sensitivity runs, and the one-factor tables.
Its "Choices made" section lists where the run departs from or fills in the
spec. **STOP** for Juan's review.

### 4. The lock

`LOCKED_HYPOTHESES_DRAFT.md` is drafted from Phase A, with its numbers
generated by the Phase A code. Once Juan approves it, it's renamed to
`LOCKED_HYPOTHESES.md`, its JSON status is set to `locked`, and it's
committed. That commit is the lock. From then on `config.py`, `pull.py`,
`features.py` and `analyze.py` are frozen: Phase B refuses to run if any of
them differs from the lock commit.

### 5. Phase B

```bash
python analyze.py --phase confirm   # once, ever
```

Before it reads anything from 2025-26, it refuses if:

- `reports/confirm_2025-26.md` already exists
- the lock file is missing, uncommitted, modified, or not `locked`
- any of the four code files changed since the lock commit

Then it runs the test suite, fits only the three locked joint models, applies
the §7 confirmation rule, and writes `reports/confirm_2025-26.md` with the
lock's commit hash.

## Layout

```
config.py              every constant (seasons, thresholds, floors, shrinkage k)
pull.py                step 1: LeagueGameLog pulls + section 4 sanity checks
features.py            step 2: baselines, outcomes, pre-game features (section 5)
analyze.py             step 3: Phase A (section 6); step 5: Phase B (sections 7-8)
conftest.py            lets pytest import the study modules from any directory
tests/                 pytest: test_pull_sanity.py (step 1), test_features.py (step 2),
                       test_confirm.py (Phase B rule, lock parsing, refusals)
data/raw/              committed parquet pulls + manifest.json (sha256 per file)
data/features/         built by features.py; git-ignored
reports/               pull_sanity.md, step1_findings.md, explore_2024-25.md + figures/
```

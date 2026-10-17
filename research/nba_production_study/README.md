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
| 2 | `features.py` + leakage tests | Waiting on Juan's neutral-site decision (see the findings) |
| 3 | Phase A: explore 2024-25 | Not started. **Stops for Juan's review.** |
| 4 | `LOCKED_HYPOTHESES.md` | Not started. Juan approves and commits it. |
| 5 | Phase B: confirm 2025-26, once | Not started |

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

### 2. Features and tests (to be built)

`features.py` builds the §5 definitions, and `tests/` gets the §9 leakage tests.
All tests must pass before any report is written.

### 3. Phase A (to be built)

`python analyze.py --phase explore` writes `reports/explore_2024-25.md`.
**STOP** for Juan's review.

### 4. The lock

`LOCKED_HYPOTHESES.md` is drafted from Phase A, then edited, approved, and
committed by Juan.

### 5. Phase B (to be built)

`python analyze.py --phase confirm` runs once. It refuses to run without a
clean, committed lock file, and it refuses if `reports/confirm_2025-26.md`
already exists.

## Layout

```
config.py              every constant (seasons, thresholds, floors, shrinkage k)
pull.py                step 1: LeagueGameLog pulls + section 4 sanity checks
conftest.py            lets pytest import the study modules from any directory
tests/                 pytest; test_pull_sanity.py covers step 1
data/raw/              committed parquet pulls + manifest.json (sha256 per file)
data/features/         step 2
reports/               pull_sanity.md now; Phase A/B reports later
```

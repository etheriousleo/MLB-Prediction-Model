# NBA closes — external scheduler (option 1)

## Why

GitHub's `schedule` trigger is not a 30-minute cron for this repo. Between Sep 27 and
Oct 9, 2026 the NBA workflow (`17,47` across 13 UTC hours = 26 scheduled runs) fired **2–4
times a day**, and the MLB workflow (`0,30`) showed the same pattern. `scripts/
snapshot_closes_nba.py` only spends a credit when a game tips within 35 minutes, so with
runs hours apart most tips are never inside a window and no close is captured.

Measured on the MLB data branch for the same reason: of 50 MLB "closes" captured since
Sep 27, 3 were quoted within 35 minutes of first pitch; the median quote was ~10 hours
early.

The fix is to supply the cadence from outside GitHub: an external scheduler calls the
workflow's `workflow_dispatch` endpoint every 30 minutes. The script's logic is unchanged.

## Setup (cron-job.org free tier, or any scheduler that can POST with headers)

0. **Prerequisite: this branch must be merged to `main` first.** A dispatch on
   `ref: main` runs `main`'s copy of the workflow file, which checks out `data` as the write
   target and fetches `scripts/snapshot_closes_nba.py` and `nba_teams.py` from `origin/main`.
   Until the merge, a dispatch runs the OLD yml and the OLD script (regular key only, no
   `mkdir -p`). The schedule fallback also only fires from `main`. Never test the cadence by
   dispatching another ref: that runs that ref's yml with `main`'s script.
1. **Token.** GitHub → Settings → Developer settings → Fine-grained personal access
   tokens → Generate. Repository access: **only** `etheriousleo/MLB-Prediction-Model`.
   Permissions: **Actions: Read and write** (Metadata: read is added automatically).
   Set an expiry you will remember to renew. Never commit it anywhere.
2. **Job.**
   - URL: `https://api.github.com/repos/etheriousleo/MLB-Prediction-Model/actions/workflows/odds_close_snapshot_nba.yml/dispatches`
   - Method: `POST`
   - Headers:
     - `Authorization: Bearer <token>`
     - `Accept: application/vnd.github+json`
     - `X-GitHub-Api-Version: 2022-11-28`
     - `Content-Type: application/json`
   - Body: `{"ref":"main"}`
   - Schedule: minutes **17 and 47**, hours **7 through 23**, time zone
     **America/New_York** (the scheduler handles DST; the script decides on the ET
     clock anyway). Runs with no tip inside the window exit without spending a credit,
     so the wide window costs ~20 seconds of Actions runner time each, nothing more.
   - Expected response: HTTP `204 No Content`.
3. **Verify.** Actions → "Snapshot NBA closing lines" should show `workflow_dispatch`
   runs every 30 minutes. On the `data` branch, `odds_closes/nba/<ET date>.json` fills
   with quotes whose `quoted_at` is within 35 minutes of each game's tip, and
   `odds_closes/nba/_credits.json` tracks the remaining credits.
4. **MLB — not yet.** `scripts/snapshot_closes.py` spends 1 credit on EVERY run with no
   tip-window gate, so a 30-minute cadence would cost ~34 credits a day (~1,000 a month) on
   the key shared with the NBA cron and both apps' prefill. Give the MLB script the same
   window gate first (an MLB change, decided separately); until then leave the MLB workflow
   on GitHub's schedule.

GitHub's own `schedule` stays in the workflow as a free fallback. Overlapping runs are
serialized by the workflow's `concurrency` group (GitHub keeps one run pending per group
and shows an older pending duplicate as "Cancelled" — that is the dedup working, not a
failure), and the script never spends twice on a slot whose close is already on file.
A non-200 from The Odds API fails the run (red in the Actions tab) except an unknown
preseason sport key, which is a quiet no-op.

## Preseason

Preseason games live under The Odds API's `basketball_nba_preseason` key (the regular key
returned nothing on Oct 4–6 while ESPN had games in the window). The script and the app
switch to that key automatically when every game on the ET date is preseason, so the
shakedown can exercise prefill, closes and CLV before Oct 20. If the key is wrong the API
answers 4xx, which is logged and not charged.

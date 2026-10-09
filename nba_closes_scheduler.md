# NBA closes — external scheduler (option 1)

## Why

GitHub's `schedule` trigger is not a 30-minute cron for this repo. Between Sep 27 and
Oct 9, 2026 the NBA workflow (`17,47` at 16 hours a day = 26 scheduled runs) fired **2–4
times a day**, and the MLB workflow (`0,30`) showed the same pattern. `scripts/
snapshot_closes_nba.py` only spends a credit when a game tips within 35 minutes, so with
runs hours apart most tips are never inside a window and no close is captured.

Measured on the MLB data branch for the same reason: of 50 MLB "closes" captured since
Sep 27, 3 were quoted within 35 minutes of first pitch; the median quote was ~10 hours
early.

The fix is to supply the cadence from outside GitHub: an external scheduler calls the
workflow's `workflow_dispatch` endpoint every 30 minutes. The script's logic is unchanged.

## Setup (cron-job.org free tier, or any scheduler that can POST with headers)

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
4. **MLB (optional, zero code change).** A second job with the same headers and body
   against `.../actions/workflows/odds_close_snapshot.yml/dispatches` at minutes 3 and 33
   gives the MLB closes the same real cadence. No MLB file is touched by this.

GitHub's own `schedule` stays in the workflow as a free fallback. Overlapping runs are
serialized by the workflow's `concurrency` group, and the script never spends twice on a
slot whose close is already on file.

## Preseason

Preseason games live under The Odds API's `basketball_nba_preseason` key (the regular key
returned nothing on Oct 4–6 while ESPN had games in the window). The script and the app
switch to that key automatically when every game on the ET date is preseason, so the
shakedown can exercise prefill, closes and CLV before Oct 20. If the key is wrong the API
answers 4xx, which is logged and not charged.

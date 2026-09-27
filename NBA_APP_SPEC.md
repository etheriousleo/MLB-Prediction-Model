# NBA App — Build Spec (v1.0)

**Repo:** `etheriousleo/MLB-Prediction-Model`  
**New files:** `nba_app.py`, `nba_pick_log.json` (data branch), `odds_closes/nba/<ET date>.json` (data branch), `nba_preseason_totals_2026.json` (main), `scripts/snapshot_closes_nba.py` + workflow  
**Freeze tag before opening night:** `nba-v1.0-frozen-<date>`  
**Season:** regular season Oct 20, 2026 – Apr 11, 2027. Preseason runs early–mid October. All-Star break Feb 19–21, 2027.

This document is the design contract for the fourth sibling app. `CLAUDE.md` governs *how* work is done in this repo; this file governs *what* gets built. Where the two conflict, `CLAUDE.md` wins on process and this file wins on design.

---

## 0. Decisions already made (do not reopen)

| Question | Decision |
|---|---|
| Parent app | Derive from **`mlb_app.py` (v4.2 market gate)** — daily-slate plumbing. Model math from **`nfl_app.py`** — margin-based. |
| What produces a real bet | **The market gate only**, from opening night. Spread and moneyline both. |
| What the model does | Runs on every game, shown as calibrated winner-confidence context, and its model-vs-price picks are logged as **paper** — never bet in v1.0. |
| Primary market | Spread. Moneyline surfaced when the market gate finds an edge (expect these on dogs/pick'ems). |
| Same-game rule | Spread + ML on the same side of the same game = **one position**, not two. Tracker enforces. |
| Prior | Preseason **win totals** (market-derived), entered once by Juan. Fallback: prior-season MOV regressed 40% to zero. |
| Fade | Linear over **15 games** per team. |
| Recent form | Last **15** games, opponent-adjusted, **30%** weight with the double-count correction. |
| Rest | **−1.5 pts** to a team on the second night of a back-to-back. Nets to 0 when both are. |
| Odds API budget | Free tier, **500 credits/month**. Target ≤ 350/month. Schedule-aware cron, both markets. |
| Model promotion | Pre-registered look at **200 paper picks per bucket** or **Feb 19, 2027**, whichever first. Verdict is CLV. Pass → model gate is *added* beside the market gate. Never model alone. |

---

## 1. Suite conventions this app inherits (non-negotiable)

Everything in `CLAUDE.md` plus, explicitly:

- **Derive, don't rebuild.** `git clone` the repo (not the Contents API — it rate-limits) and derive `nba_app.py` from the live `mlb_app.py`. Shared patterns must propagate, not be reconstructed.
- **Discipline is structural.** Thresholds hardcoded. No sliders or UI controls that could loosen any criterion. No backtest tab.
- **ET clock helper everywhere.** Streamlit Cloud runs UTC. Every date, cache key, log row, and cron decision uses the ET helper. Note DST ends Nov 1, 2026 — the cron window must be expressed so the ET helper, not the cron string, decides.
- **Identify games by ESPN event ID**, never by matchup string. All team joins by ESPN team ID.
- **Data branch persistence.** Pick log and closes live on `data`; commits to `main` reboot the app.
- **Own Streamlit Secrets** for the new app (`[odds]` key, GitHub token). Repo Actions secret `ODDS_API_KEY` is shared.
- **Widget key namespace** is fresh (`nba_` prefix, versioned) — no session-state carryover.
- **Flash-message pattern** for button reports.
- **Every game on the slate is logged**, bet or not. Un-bet games are the control group. Graded rows are immutable; ungraded rows from a superseded version are replaced on the next Log click.
- **Model version named and frozen** in code with a comment prohibiting parameter changes before the pre-registered look.
- **Do not touch `mlb_app.py` or the MLB cron/workflow.** The MLB postseason is live through October.

---

## 2. Data sources

### ESPN (free, no key)
- Scoreboard: `site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD` — supports range queries `dates=YYYYMMDD-YYYYMMDD`. **Use the WNBA app's range-query pattern**: one season-to-date range pull gives every game → games played, MOV, last-15, opponent adjustment, and yesterday's games for back-to-back detection. Avoid per-team schedule calls.
- Standings: `site.api.espn.com/apis/v2/sports/basketball/nba/standings` (`?season=2026` for prior season — carries point differential; only needed for the fallback prior).
- Season type from each event's `season.type`: 1 = preseason, 2 = regular, 3 = postseason. Play-in may carry its own type — **verify against last April's feed via `dates=`** before relying on it.
- Neutral-site flag on the event (NBA Cup final, international games).

### The Odds API
- Sport key `basketball_nba`, markets `h2h,spreads`, `bookmakers=fanduel,pinnacle,<2–3 consensus books>`. Cost = markets × 1 = **2 credits per pull** with the bookmakers param.
- Feed includes in-play prices — **filter by commence time** as MLB does.
- Devig two-sided per book: `BE_pick / (BE_pick + BE_opp)`.
- App reads and displays the `x-requests-remaining` header so the burn is visible.

---

## 3. Model (frozen as `nba-v1.0`)

All constants hardcoded. Comment block at the top of the model section: version, freeze date, "no parameter changes before the pre-registered look."

### 3.1 Team strength (points of margin)

```
HFA          = 2.5      # points, home
SIGMA        = 12.5     # SD of game margin
FADE_GAMES   = 15
FORM_WINDOW  = 15
FORM_WEIGHT  = 0.30     # nominal; corrected below
B2B_PENALTY  = 1.5      # points
PRIOR_REGRESS = 0.60    # fallback prior only: prior-season MOV × 0.60
```

**Prior (per team, before any games):**
- Primary: `prior_margin = (win_total − 41) / 2.7` from `nba_preseason_totals_2026.json` (30 teams, entered by Juan from FanDuel's preseason win totals).
- Fallback if a team is missing from the file: prior-season MOV × `PRIOR_REGRESS` from ESPN standings `?season=2026`.

**Season-to-date margin:** average MOV over regular-season games only (exclude preseason, exclude the NBA Cup final), **opponent-adjusted** with the same SRS-style iteration as `cfb_app.py`.

**Recent form:** last `FORM_WINDOW` games, opponent-adjusted the same way. Effective weight uses the double-count correction from the MLB lesson:
```
share  = min(FORM_WINDOW, n) / n
w_int  = max(0, (FORM_WEIGHT − share) / (1 − share))   # → 0 while n ≤ FORM_WINDOW
season_blend = (1 − w_int) · season_adj + w_int · form_adj
```

**Fade:**
```
w_prior  = max(0, 1 − n / FADE_GAMES)
strength = w_prior · prior_margin + (1 − w_prior) · season_blend
```

### 3.2 Game projection

```
rest_adj = B2B_PENALTY · (away_on_b2b − home_on_b2b)      # positive favors home
hfa      = 0 if neutral_site else HFA
M        = (strength_home − strength_away) + hfa + rest_adj   # projected home margin
P_home_win   = Φ(M / SIGMA)
P_home_cover(L) = Φ((M − L) / SIGMA)     # L = −home_spread (points home must win by)
```

Back-to-back = the team played on the previous ET date (regular season or preseason game both count). No 3-in-4, no travel, no injuries in v1.0 — injuries are the #1 candidate for the tuning window, not now.

Single distribution parameter (`SIGMA`) for both win and cover probability. No separate `K_MARGIN` — the CFB look found cover probabilities overclaimed under a second scale; don't reintroduce it.

**Display tiers** (context only, reuse MLB thresholds on the model's pick): High ≥ 65%, Moderate ≥ 56.5%.

---

## 4. Gates

Outputs are the suite's three verdicts, hardcoded: **GOOD PRICE / THIN — NO BET / STAY AWAY**.

### 4.1 Market gate (live from Oct 20 — the only thing that produces a bet)

Compares FanDuel to the sharp reference (Pinnacle; consensus of the other non-FanDuel books as fallback). Threshold **≥ 1.5pp**, same as MLB v4.2.

**Moneyline:** `fair_p` = Pinnacle devigged prob for the side; `BE` = FanDuel single-sided break-even for that side. `edge = fair_p − BE`. Both sides evaluated.

**Spread** (sign convention: home spread `s`, negative = home favored; `L = −s`):
1. Pinnacle: home spread `s_P`, devigged `p_P` = P(home covers). Fair margin `M* = L_P + SIGMA · Φ⁻¹(p_P)`.
2. FanDuel home side at `s_F`, price `a_F`: `BE_F` single-sided from `a_F`. `fair_p = Φ((M* − L_F) / SIGMA)`. `edge = fair_p − BE_F`.
3. Away side symmetric (`M*` negated, away line = `+s_F`).
4. Consensus fallback: average `M*` across available books.

Verdict per side per market: GOOD PRICE if `edge ≥ 0.015`; THIN if `0 ≤ edge < 0.015`; STAY AWAY otherwise.

FanDuel prices/lines **prefill from the feed** (MLB v4.2 pattern). Manual override field stays available for a line FanDuel shows that the feed missed, but the logged row records which source was used.

### 4.2 Paper gate (model-vs-price — logged, never bet in v1.0)

Model's pick side vs FanDuel break-even, cushion **≥ 3pp** (same as CFB/NFL):
- `paper_ml`: pick = side with `P_win > 0.5`; `cushion = P_win − BE_FD`.
- `paper_ats`: pick = side with `P_cover > 0.5` at FanDuel's line; `cushion = P_cover − BE_FD`.

Same three verdict labels, stored under `paper_*` fields. **No bet button is wired to these.** A paper pick **counts toward the pre-registered look only when both teams have `w_prior = 0`** (≥ 15 regular-season games) — store the flag at log time.

### 4.3 Position rule

If the market gate says GOOD PRICE on both the spread and ML of the same side, the tracker records one position (default: the spread; ML noted). Never two stakes on one side of one game.

---

## 5. Odds feed and closing-line cron

### Prefill (on app open / Refresh)
One pull, both markets, **2 credits**. Cached by ET date + hour. Expected 30–60 credits/month.

### Closing lines — `scripts/snapshot_closes_nba.py`
Generalize `scripts/snapshot_closes.py` with a sport argument **only if it's a clean parameterization that leaves the MLB path byte-identical**; otherwise a sibling script. Either way, a separate workflow file for NBA.

Mechanics (this is the credit-saving change vs MLB):
1. Workflow cron fires **every 30 min at :17 and :47** (off-peak minutes — Actions delays runs most at :00/:30), across a UTC window wide enough to cover 12:30 pm–11:00 pm ET in both EDT and EST.
2. Script pulls the ET date's scoreboard from ESPN (free), builds the set of distinct tip times.
3. **Calls the Odds API only if** some game tips within the next 35 minutes *and* no close has been captured for that tip slot yet. Otherwise exits without spending a credit.
4. Snapshot = both markets, all bookmakers above, **2 credits**. Written to `odds_closes/nba/<ET date>.json` keyed by ESPN event ID; last pre-tip quote for each game is the close.
5. Hard reserve: if `x-requests-remaining < 40`, skip and log a warning (keeps prefill alive at month end).

Budget math: ~5 distinct slots on a typical weeknight × 2 = 10 credits; weekends ~8 slots × 2 = 16; ≈ 300/month + prefill. Under 350 with headroom. The close is ~13 minutes pre-tip on this cadence — late injury moves may be missed; accepted for the free tier.

Auto-grade fills `close_spread`, `close_spread_price`, `close_ml` into the row and computes CLV: **ATS in spread points, ML in probability points** (NFL pattern).

---

## 6. Pick tracker

Every regular-season and postseason game gets a row. Preseason games are **not logged** (they're for shakedown only — see §8).

Row fields (extend the MLB row; don't rename existing shared fields):
`espn_event_id, date_et, tip_et, home_id, away_id, season_type, neutral_site, cup_final, home_b2b, away_b2b, model_version, model_margin, p_home_win, fd_spread, fd_spread_price, fd_ml_home, fd_ml_away, ref_book, ref_spread, ref_spread_price, ref_ml_home, ref_ml_away, fair_margin, edge_ats_home, edge_ats_away, edge_ml_home, edge_ml_away, verdict_market_ats, verdict_market_ml, verdict_paper_ats, verdict_paper_ml, paper_counts (bool), bet (bool), bet_market, bet_side, bet_line, bet_price, stake, close_spread, close_spread_price, close_ml_home, close_ml_away, clv_ats_pts, clv_ml_pp, final_home, final_away, ot (bool), result_ats, result_ml, pnl, graded_at`

Grading rules:
- No ties in the NBA. OT results count normally.
- ATS push on a whole-number line: stake returned, excluded from W-L, CLV still recorded.
- **NBA Cup final** (Dec 11): logged and gradeable as a bet, but `cup_final = True` and it is **excluded from all team-strength stats** (it doesn't count in standings).
- Postponed: row stays ungraded with a `postponed` flag; re-grades when ESPN reschedules under the same event ID.
- Postseason (play-in + playoffs): supported like MLB's postseason mode; `season_type` recorded.

Buckets in the summary view (each with W-L, units, CLV, n): `market_ats`, `market_ml`, `paper_ats`, `paper_ml`. Paper buckets show both "all" and "counts" (post-fade) totals.

Real-money fields: `STAKE_UNITS` and `SEASON_BUDGET` set in code before the first bet — **Juan sets the values**; no UI reload. Stake column drives P/L against the budget.

---

## 7. Pre-registered evaluation (record this in code comments and in `nba_evaluation_plan.md`)

**When:** the earlier of 200 `paper_counts` picks in a bucket, or Feb 19, 2027. **Look once.** Spread and ML are evaluated as separate buckets.

**Pass criteria (per bucket):**
- Mean CLV ≥ +0.5 spread points (`paper_ats`) or ≥ +1.0pp (`paper_ml`), with the 95% interval excluding zero.
- Record does not contradict (ATS ≥ 48%; ML units not below −5% per bet).

**If pass:** that bucket's model gate is enabled as a *second* bet source alongside the market gate, at reduced stake (0.5 × `STAKE_UNITS`) for the post-break regular season (~6 weeks, ~400 games). Re-version to `nba-v1.1`.  
**If fail:** stays paper; nothing changes.  
**Never:** model gate alone. The market gate keeps running in all cases.

Win rate over 200 picks has SE ≈ 3.5pp — it cannot acquit or convict the model. CLV is the verdict; the record is a sanity check.

---

## 8. Build and shakedown sequence

1. Clone. Read `CLAUDE.md`. Read this file. Read `mlb_app.py`, `nfl_app.py`, `cfb_app.py` (SRS block), the WNBA app's ESPN range-query code, and `scripts/snapshot_closes.py`.
2. Derive `nba_app.py`. Keep the MLB file structure and section order so future fixes propagate.
3. Build the `AppTest` harness against a deterministic mock ESPN + mock Odds feed covering: overtime game; back-to-back for home / away / both; neutral-site game; postponed game; preseason game on the slate (must be excluded from stats and log); a 1 pm weekend tip; a day with no games; in-play prices in the feed (must be filtered); Pinnacle missing (consensus fallback); whole-number spread push; NBA Cup final flag; both-markets-GOOD same side (position rule).
4. Deploy as a new Streamlit Cloud app with its own Secrets. Add the NBA cron workflow.
5. **Preseason shakedown (early–mid Oct):** run the app against live preseason slates to verify ESPN parsing, tip-time detection, cron credit behavior, closes landing on the data branch, and auto-grade. Nothing from preseason enters the log or the stats.
6. Before Oct 20: Juan enters the 30 win totals and sets `STAKE_UNITS` / `SEASON_BUDGET`. Tag `nba-v1.0-frozen-<date>`.
7. Oct 20: market gate live. Paper gate logging. Fade running.

---

## 9. Explicitly out of scope for v1.0

- Injury / lineup data of any kind (the market gate carries it; revisit in the tuning window).
- 3-in-4 and travel adjustments.
- Totals (over/under).
- Player props, parlays, live betting.
- Any slider, threshold control, or backtest.
- Any change to the MLB, CFB, or NFL apps or workflows.

---

## 10. Open items — Juan supplies

- [ ] 30 preseason win totals (FanDuel) → `nba_preseason_totals_2026.json`
- [ ] `STAKE_UNITS`, `SEASON_BUDGET`
- [ ] Streamlit Cloud app created; Secrets populated
- [ ] Confirm which consensus books to include alongside Pinnacle (suggest: DraftKings, BetMGM, Caesars)

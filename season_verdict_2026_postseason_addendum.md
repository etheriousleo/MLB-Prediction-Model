# 2026 Postseason Addendum — v4.2 market gate (Sept 27 – Oct 28)

Descriptive only. No pre-registration existed for this window; nothing here
overrides the binding regular-season verdict (`season_verdict_2026.md`).

**Window:** 55 rows, 52 graded (14 regular-season tail, 38 postseason), 47 priced,
51 with a closing line, 5 hand-entered overrides (boosts).

## Market gate (FanDuel price on the model's pick vs Pinnacle fair line)
- GOOD PRICE fired **0 times on feed prices**. Across 45 feed-priced picks the
  FanDuel-vs-Pinnacle gap ranged −3.7 to −0.6pp, median −2.1pp (≈ the vig).
  A single soft book cannot beat the sharp fair line on the same side; the
  gate can only say yes to promotions. Structural, not sample-size.
- Promo overrides: 4 boosts, 2–2, +0.12u. Uninformative at n=4.
- Stay-aways: 29–13 (69%), +9.34u — outcomes, see below.

## Closing-line value (excl. overrides)
- 41 picks: beat the close 18/41 (44%), **avg CLV −0.96pp** — the market moved
  against the model's picks. Direction consistent with the regular-season verdict.

## Model (all v4.2 picks)
- 35–17 (67%), +7.46u flat; postseason 24–14, regular-season tail 11–3.
  ~1 SD of noise at n=52 and mostly out-of-distribution games; CLV is the
  more reliable signal here and it is negative.
- Tiers did not separate (High 19–8, Low 14–5, Conflicted 2–4).
- Form diagnostic: one row — at 15% effective weight the component moved no
  pick beyond ±1.7pp; the ≥2pp instrument cannot answer the form question.

## Infrastructure
- Closing capture 51/52 after the backfill fix; override persistence, loss-limit
  accounting, A's name canonicalization, live-price exclusion all in production.

## Carry-forward (decide in January with the CFB post-bowls verdict in hand)
1. Market gate as built (one book, moneylines) is inert — retire or rebuild on
   best-price-across-books.
2. v5 candidate inputs the market prices late: confirmed lineups, bullpen usage.
3. Calibration map (slope 1.81) as the 2027 probability display, validated forward.
4. Form: replay over 2026 snapshots (0/15/35) or diagnostic threshold → 1pp.

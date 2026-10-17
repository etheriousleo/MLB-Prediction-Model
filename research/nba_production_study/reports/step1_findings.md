# Step 1 findings: pull and sanity checks

Spec §11.1 stop point. The pull ran on Juan's machine on 2026-10-16 with
nba_api 1.11.4. Everything here is outcome-free: row counts, data structure,
and schedule facts. Nothing describes how any player performed.

## Row counts

| Season | Role | Player rows | Team rows | Games |
|---|---|---|---|---|
| 2024-25 | explore | 26,306 | 2,460 | 1,230 |
| 2025-26 | confirm | 26,651 | 2,460 | 1,230 |

The §4 checks pass for both seasons (see `pull_sanity.md`): 1,230 games,
30 teams at 82 games each, no NBA Cup final, no duplicate rows, the player
and team logs cover the same team-games, and player points sum to team
points in every team-game. That holds after one amendment, described below.

## Non-playing players: not listed

A player who did not play has **no row** in the player log. The evidence
from 2024-25:

- No row has MIN null.
- 100 rows have MIN 0, spread over 79 of 2,460 team-games. A DNP listing
  would add about 2-5 rows to every team-game, which is thousands.
- MIN is recorded in whole minutes, and the counts rise smoothly from
  0 minutes (100 rows) to 1 (390) and 2 (594). The MIN-0 rows are stints
  shorter than 30 seconds, rounded down. 20 of them even have a stat.

What this means for step 2:

- A player "played" when he has a row with MIN > 0. That matches the §5.1
  unit, and it makes fresh-absence detection (§5.3) a question of whether
  a MIN > 0 row exists.
- A regular whose only row tonight is a MIN-0 cameo counts as absent. He
  contributed nothing, which is the point of the feature. This should be
  rare: a 20-minute player who logs under 30 seconds was almost certainly
  hurt at the start of the game.
- Per-minute rates use whole minutes. With the MIN >= 10 floor on
  `rate_resid_S`, rounding moves a rate by at most about 5%.

## Neutral-site games: an amendment to the §4 check

The spec requires one home row ("vs.") and one away row ("@") per game.
Five games per season fail that rule. In each of them, **both** rows carry
"@", which is how stats.nba.com marks a neutral site:

| Season | GAME_ID | Venue |
|---|---|---|
| 2024-25 | 0022400147 | Mexico City, Nov 2 (MIA-WAS) |
| | 0022400621, 0022400633 | Paris, Jan 23 and 25 (IND-SAS) |
| | 0022401229, 0022401230 | Las Vegas, NBA Cup semifinals, Dec 14 |
| 2025-26 | 0022500147 | Mexico City, Nov 1 (DAL-DET) |
| | 0022500578 | Berlin, Jan 15 (MEM-ORL) |
| | 0022500602 | London, Jan 18 (MEM-ORL) |
| | 0022501229, 0022501230 | Las Vegas, NBA Cup semifinals, Dec 13 |

The check now accepts a game with "@" on both rows only if it is listed in
`config.NEUTRAL_SITE_GAMES`. Any other anomaly still fails loudly, and so
does a listed game that turns out to have a home team.

### Open decision for Juan: how the analysis treats these games

The spec doesn't anticipate neutral sites, and they break two definitions.
`is_home` would score both teams as away, and `arena_team_id` names the home
team as a stand-in for the scorekeeper, but neither team's crew works a
game in Berlin.

**Proposed rule:**

- **Keep** these games in every backward-looking calculation: baselines,
  rest days and back-to-backs, team and opponent rates, last-5 form, and
  fresh absences. They are real games. Dropping them would, for example,
  make the Pacers look rested after Paris.
- **Exclude** them as analysis rows (the outcome units), because there is
  no home team and no home scorekeeper. That costs about 0.4% of rows
  (5 of 1,230 games).

The alternative is to keep them as analysis rows with `is_home = 0` for both
teams and a separate "neutral" arena. That adds a third category to the home
effect for about 10 games, so it isn't recommended.

**Status:** awaiting Juan's decision. Step 2 (`features.py`) starts once he
decides.

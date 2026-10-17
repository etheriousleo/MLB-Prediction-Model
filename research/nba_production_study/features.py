"""
Step 2 of NBA_STUDY_SPEC.md (§5): baselines, outcomes, and pre-game features
for every player-game.

    python features.py      # builds data/features/2024-25.parquet (explore only)

The confirm season is never built from here. Only `analyze.py --phase
confirm` builds it, after the lock (§7-8); see build_season.

No look-ahead (§2). Every pre-game quantity for a game on date D uses games
on dates strictly before D. A player plays at most once per date, and so
does a team, so "earlier games" and "earlier dates" are the same thing here,
and build_features asserts it. tests/test_features.py enforces the rule by
randomizing every stat after a cutoff date and checking that nothing on or
before it moves.

The one deliberate exception is the spec's own (§5.3): fresh absences use
tonight's actual box score to see who sat. A late scratch isn't knowable
before tip-off; that's accepted for a drivers study and stated in the report.

Choices the spec leaves open, each also commented where it's made:
- A player "played" when he has a row with MIN > 0. Step 1 found that DNPs
  aren't listed and MIN-0 rows are sub-30-second cameos
  (reports/step1_findings.md). MIN-0 rows are dropped first, so they are
  neither units nor prior games, and a regular whose only row tonight is
  MIN 0 counts as absent.
- Neutral-site games (config.NEUTRAL_SITE_GAMES) count in every backward-
  looking calculation but are never analysis units (Juan, 2026-10-17).
- Team and opponent rates are pooled season-to-date ratios (for example
  100 * sum of points allowed / sum of possessions), not means of per-game
  rates, shrunk toward the league's pooled value as of the same date.
- The denominators of the miss_*_share features are the team's raw
  season-to-date averages per game. They aren't rate features, so they
  aren't shrunk.
- A teammate traded away still counts as a fresh absence for his old team
  until he appears for his new one. His production has left the team, and
  that is exactly what the feature measures.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
import pull

STATS = config.STATS
K = config.SHRINK_K

# --- Output layout. PREGAME_COLUMNS is what the no-look-ahead tests compare. ---
ID_COLUMNS = ["SEASON", "GAME_ID", "GAME_DATE", "PLAYER_ID", "TEAM_ID", "OPP_TEAM_ID", "stint"]
BASELINE_COLUMNS = ["n_prior", "base_min",
                    *[f"{kind}_{s}" for s in STATS for kind in ("base", "sd", "base_rate")]]
FEATURE_COLUMNS = [
    # context
    "is_home", "days_rest", "b2b", "opp_b2b", "three_in_four", "season_phase",
    # opponent
    "opp_pace", "opp_drtg", "opp_efg_allowed", "opp_ast_allowed",
    "opp_fg_pct", "opp_3pa_rate", "opp_oreb_pct", "opp_dreb_pct",
    # teammates (fresh absences)
    "n_fresh_absent", "miss_usage_share", *[f"miss_{s}_share" for s in STATS],
    # game script
    "net_gap", "abs_net_gap",
    # form
    *[f"last5_dev_{s}" for s in STATS], "last5_min_dev",
    # arena
    "arena_team_id",
]
FLAG_COLUMNS = ["neutral_site", "eligible", *[f"elig_{s}" for s in STATS]]
PREGAME_COLUMNS = BASELINE_COLUMNS + FEATURE_COLUMNS + FLAG_COLUMNS
OUTCOME_COLUMNS = ["MIN", *STATS, *[f"z_{s}" for s in STATS], *[f"hit_{s}" for s in STATS],
                   "min_resid", *[f"rate_resid_{s}" for s in STATS]]
DIAGNOSTIC_COLUMNS = ["ot_game", "margin", "early_exit"]   # §5.4: never driver inputs
OUTPUT_COLUMNS = ID_COLUMNS + PREGAME_COLUMNS + OUTCOME_COLUMNS + DIAGNOSTIC_COLUMNS

# Team rates (§5.3) as pooled ratios: name -> (numerator, denominator, scale).
# Each becomes a season-to-date value shrunk toward the league (see team_pregame).
RATE_DEFS = {
    "pace":        ("POSS", "GAME_MIN", 48.0),     # possessions per 48 minutes
    "drtg":        ("OPP_PTS", "POSS", 100.0),     # points allowed per 100 possessions
    "net":         ("NET_PTS", "POSS", 100.0),     # net rating: (PTS - OPP_PTS) per 100
    "efg_allowed": ("OPP_EFG_MADE", "OPP_FGA", 1.0),  # defensive eFG%
    "ast_allowed": ("OPP_AST", "POSS", 100.0),     # assists allowed per 100 possessions
    "fg_pct":      ("FGM", "FGA", 1.0),            # own shooting
    "3pa_rate":    ("FG3A", "FGA", 1.0),           # own 3PA / FGA
    "oreb_pct":    ("OREB", "OREB_CHANCES", 1.0),  # OREB / (OREB + opponent DREB)
    "dreb_pct":    ("DREB", "DREB_CHANCES", 1.0),  # DREB / (DREB + opponent OREB)
}
# The opponent's value of each of these becomes an opp_* feature.
OPPONENT_RATES = ("pace", "drtg", "efg_allowed", "ast_allowed",
                  "fg_pct", "3pa_rate", "oreb_pct", "dreb_pct")

TEAM_STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "PTS"]


def usage(df: pd.DataFrame) -> pd.Series:
    """FGA + 0.44*FTA + TOV: the usage measure in miss_usage_share (§5.3)."""
    return df["FGA"] + config.POSS_FTA_COEF * df["FTA"] + df["TOV"]


# ---------------------------------------------------------------------------
# Player side: stints, baselines, form
# ---------------------------------------------------------------------------

def _sum_before(values, keys):
    """Running sum within each group over the rows before each row (0 for
    the first). The row itself is shifted out first, never subtracted later."""
    return values.groupby(keys).shift(fill_value=0).groupby(keys).cumsum()


def player_games(player: pd.DataFrame) -> pd.DataFrame:
    """Rows where the player got on the floor, sorted by player and date."""
    cols = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "FGA", "FTA", "TOV", *STATS]
    pl = player.loc[player["MIN"] > 0, cols].copy()   # MIN-0 cameos are not games (see docstring)
    if pl.duplicated(["PLAYER_ID", "GAME_DATE"]).any():
        raise ValueError("a player has two games on one date, so 'prior games' would be ambiguous")
    pl["USG"] = usage(pl)
    return pl.sort_values(["PLAYER_ID", "GAME_DATE"], ignore_index=True)


def add_baselines(pl: pd.DataFrame) -> pd.DataFrame:
    """Stints, prior-game baselines (§5.2) and last-5 form (§5.3).

    A stint is a run of consecutive games for one team. It resets whenever the
    player changes teams, whether by trade, waiver, or signing. Every prior
    quantity covers earlier games in the same stint only: each running sum is
    taken over the shifted series, so the current game never enters its own
    baseline, not even as an add-then-subtract. (That would leak in the last
    floating-point bits; the same-day perturbation test caught it.) The
    incl_* columns do include the current game. They describe a player as of
    the morning after, which is what teammate absences need."""
    new_stint = pl["TEAM_ID"].ne(pl.groupby("PLAYER_ID")["TEAM_ID"].shift())
    pl["stint"] = new_stint.astype(int).groupby(pl["PLAYER_ID"]).cumsum()
    by = pl.groupby(["PLAYER_ID", "stint"], sort=False)

    keys = [pl["PLAYER_ID"], pl["stint"]]
    n = by.cumcount()                      # prior games in the stint
    pl["n_prior"] = n
    for col in ("MIN", "USG", *STATS):
        x = pl[col].astype(float)
        prior_sum = _sum_before(x, keys)
        pl[f"prior_sum_{col}"] = prior_sum
        pl[f"prior_mean_{col}"] = (prior_sum / n).where(n > 0)
        pl[f"incl_mean_{col}"] = x.groupby(keys).cumsum() / (n + 1)
        if col in STATS:
            # Sample SD (ddof=1) of the prior games, from running sums of squares.
            prior_sq = _sum_before(x * x, keys)
            var = (prior_sq - prior_sum ** 2 / n) / (n - 1)
            pl[f"prior_sd_{col}"] = np.sqrt(var.clip(lower=0)).where(n > 1)
        if col in ("MIN", *STATS):
            # Mean of the last FORM_WINDOW prior games in the stint.
            pl[f"last5_{col}"] = by[col].transform(
                lambda s: s.shift().rolling(config.FORM_WINDOW).mean())

    pl["base_min"] = pl["prior_mean_MIN"]
    for s in STATS:
        pl[f"base_{s}"] = pl[f"prior_mean_{s}"]
        pl[f"sd_{s}"] = pl[f"prior_sd_{s}"].clip(lower=config.SD_FLOOR)   # NaN stays NaN
        pl[f"base_rate_{s}"] = (pl[f"prior_sum_{s}"] / pl["prior_sum_MIN"]).where(n > 0)
        pl[f"last5_dev_{s}"] = (pl[f"last5_{s}"] - pl[f"base_{s}"]) / pl[f"sd_{s}"]
    pl["last5_min_dev"] = pl["last5_MIN"] - pl["base_min"]
    return pl


# ---------------------------------------------------------------------------
# Team side: rest, shrunk season-to-date rates, per-game averages
# ---------------------------------------------------------------------------

def team_games(team: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per team-game with the opponent's box score alongside."""
    t = team[["GAME_ID", "TEAM_ID", "GAME_DATE", "MATCHUP", *TEAM_STATS]].copy()
    t["POSS_EST"] = t["FGA"] + config.POSS_FTA_COEF * t["FTA"] - t["OREB"] + t["TOV"]
    t["USG"] = usage(t)
    opp = (t[["GAME_ID", "TEAM_ID", *TEAM_STATS, "POSS_EST"]]
           .add_prefix("OPP_").rename(columns={"OPP_GAME_ID": "GAME_ID"}))
    tg = t.merge(opp, on="GAME_ID")
    tg = tg[tg["TEAM_ID"] != tg["OPP_TEAM_ID"]].reset_index(drop=True)
    if len(tg) != len(t):
        raise ValueError("every game needs exactly two team rows (pull.py checks this)")

    # Possessions per game: the two teams' estimates, averaged (§5.3).
    tg["POSS"] = (tg["POSS_EST"] + tg["OPP_POSS_EST"]) / 2
    tg["GAME_MIN"] = tg["MIN"] / 5                     # 240 team minutes = a 48-minute game
    tg["NET_PTS"] = tg["PTS"] - tg["OPP_PTS"]
    tg["OPP_EFG_MADE"] = tg["OPP_FGM"] + 0.5 * tg["OPP_FG3M"]
    tg["OREB_CHANCES"] = tg["OREB"] + tg["OPP_DREB"]
    tg["DREB_CHANCES"] = tg["DREB"] + tg["OPP_OREB"]

    tg["neutral_site"] = tg["GAME_ID"].isin(list(config.NEUTRAL_SITE_GAMES.get(season, {})))
    home = tg["MATCHUP"].str.contains("vs.", regex=False)
    tg["is_home"] = home.astype(float)
    # Arena = the home team, as a proxy for the scorekeeper (§5.3). A neutral
    # site has neither team's crew, so it gets no arena.
    tg["arena_team_id"] = pd.array(np.where(home, tg["TEAM_ID"], tg["OPP_TEAM_ID"]), dtype="Int64")
    tg.loc[tg["neutral_site"], "arena_team_id"] = pd.NA

    # §5.4 diagnostics: post-game, never driver inputs.
    tg["ot_game"] = (tg["MIN"] > config.REGULATION_TEAM_MIN).astype(float)
    tg["margin"] = tg["NET_PTS"]
    return tg


def _games_in_prior_days(dates: pd.Series, days: int) -> pd.Series:
    """For each game in one team's date-sorted schedule, how many of its games
    fell in the `days` days before it."""
    d = dates.to_numpy()
    lo = np.searchsorted(d, d - np.timedelta64(days, "D"), side="left")
    return pd.Series(np.arange(len(d)) - lo, index=dates.index)


def team_pregame(tg: pd.DataFrame) -> pd.DataFrame:
    """Rest flags, shrunk season-to-date rates, and per-game averages, as of
    the morning of each team-game, plus the opponent's versions."""
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"], ignore_index=True)
    if tg.duplicated(["TEAM_ID", "GAME_DATE"]).any():
        raise ValueError("a team has two games on one date")
    by = tg.groupby("TEAM_ID", sort=False)
    n = by.cumcount()                                  # team games before this one
    tg["n_team_prior"] = n

    # Rest (§5.3). days_rest = days since the previous game, minus 1.
    tg["days_rest"] = (tg["GAME_DATE"] - by["GAME_DATE"].shift()).dt.days - 1
    tg["b2b"] = tg["days_rest"].eq(0).astype(float).where(tg["days_rest"].notna())
    # Third game in four nights: two or more games in the three days before.
    tg["three_in_four"] = (by["GAME_DATE"].transform(lambda d: _games_in_prior_days(d, 3)) >= 2
                           ).astype(float)

    # Season-to-date sums over the team's earlier games, and over the whole
    # league's earlier dates. Both stop the day before, never on it.
    rate_cols = sorted({c for num, den, _ in RATE_DEFS.values() for c in (num, den)})
    sum_cols = sorted(set(rate_cols) | {"USG", *STATS})
    team_before = _sum_before(tg[sum_cols], tg["TEAM_ID"])
    daily = tg.groupby("GAME_DATE")[rate_cols].sum().sort_index()
    league_before = daily.cumsum().shift(fill_value=0).reindex(tg["GAME_DATE"]).reset_index(drop=True)

    # Shrinkage (§5.3): x = (n*team + k*league) / (n + k). On opening night
    # the league has no games yet and every rate is NaN; no such game is ever
    # eligible (10 prior games are required).
    for name, (num, den, scale) in RATE_DEFS.items():
        team_rate = scale * team_before[num] / team_before[den]
        league_rate = scale * league_before[num] / league_before[den]
        tg[name] = ((n * team_rate + K * league_rate) / (n + K)).where(n > 0, league_rate)

    # Raw per-game averages: the denominators of the miss_*_share features.
    for col in ("USG", *STATS):
        tg[f"avg_{col}"] = (team_before[col] / n).where(n > 0)

    # Each opponent's own values, joined back onto this team-game.
    opp = tg[["GAME_ID", "TEAM_ID", "days_rest", *OPPONENT_RATES, "net"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "days_rest": "opp_days_rest", "net": "opp_net",
                 **{r: f"opp_{r}" for r in OPPONENT_RATES}})
    tg = tg.merge(opp, on=["GAME_ID", "OPP_TEAM_ID"], validate="one_to_one")
    tg["opp_b2b"] = tg["opp_days_rest"].eq(0).astype(float).where(tg["opp_days_rest"].notna())
    tg["net_gap"] = tg["net"] - tg["opp_net"]
    tg["abs_net_gap"] = tg["net_gap"].abs()
    return tg


# ---------------------------------------------------------------------------
# Teammates: fresh absences (§5.3)
# ---------------------------------------------------------------------------

def fresh_absences(pl: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """For each team-game, the regulars who are freshly absent tonight.

    A regular has, before tonight, at least 10 games in his current stint
    with this team at an average of at least 20 minutes. He is freshly absent
    when he has no MIN > 0 row tonight but played in at least one of the
    team's previous 5 games. Longer absences are already baked into the
    baseline. A candidate whose latest game before tonight was for another
    team has moved on, so he doesn't count.

    Returns n_fresh_absent and the absentees' summed pre-game averages
    (miss_sum_USG, miss_sum_PTS, ...), keyed by GAME_ID and TEAM_ID.
    """
    played = pl.groupby(["TEAM_ID", "GAME_ID"])["PLAYER_ID"].agg(frozenset).to_dict()
    # pl is sorted by player then date, so each player's rows are in date order.
    rows_by_player = {pid: idx.to_numpy() for pid, idx in pl.groupby("PLAYER_ID").groups.items()}
    dates = pl["GAME_DATE"].to_numpy()
    team_of = pl["TEAM_ID"].to_numpy()
    n_incl = pl["n_prior"].to_numpy() + 1
    min_incl = pl["incl_mean_MIN"].to_numpy()
    summed = ["USG", *STATS]
    incl = np.column_stack([pl[f"incl_mean_{c}"].to_numpy() for c in summed])

    out = []
    for team_id, games in tg.sort_values("GAME_DATE").groupby("TEAM_ID", sort=False):
        gids, gdates = games["GAME_ID"].to_numpy(), games["GAME_DATE"].to_numpy()
        for j, (gid, gdate) in enumerate(zip(gids, gdates)):
            recent = set().union(*(played.get((team_id, g), ())
                                   for g in gids[max(0, j - config.FRESH_ABSENCE_LOOKBACK):j]))
            absent = recent - played.get((team_id, gid), frozenset())
            count, sums = 0, np.zeros(len(summed))
            for pid in absent:
                rows = rows_by_player[pid]
                # His latest game before tonight. One exists: he played in a recent one.
                r = rows[np.searchsorted(dates[rows], gdate, side="left") - 1]
                if team_of[r] != team_id:
                    continue                              # he has since played elsewhere
                if n_incl[r] >= config.REGULAR_MIN_PRIOR_GAMES and \
                        min_incl[r] >= config.REGULAR_MIN_BASE_MINUTES:
                    count += 1
                    sums += incl[r]
            out.append((gid, team_id, count, *sums))
    return pd.DataFrame(out, columns=["GAME_ID", "TEAM_ID", "n_fresh_absent",
                                      *[f"miss_sum_{c}" for c in summed]])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_features(player: pd.DataFrame, team: pd.DataFrame, season: str) -> pd.DataFrame:
    """Every player-game with MIN > 0, with baselines, pre-game features,
    eligibility flags, outcomes and diagnostics. Inputs come from pull.load_raw."""
    pl = add_baselines(player_games(player))
    tg = team_pregame(team_games(team, season))
    tg = tg.merge(fresh_absences(pl, tg), on=["GAME_ID", "TEAM_ID"], validate="one_to_one")
    tg["miss_usage_share"] = tg["miss_sum_USG"] / tg["avg_USG"]
    for s in STATS:
        tg[f"miss_{s}_share"] = tg[f"miss_sum_{s}"] / tg[f"avg_{s}"]

    team_cols = ["GAME_ID", "TEAM_ID", "GAME_DATE", "OPP_TEAM_ID", "neutral_site", "is_home",
                 "days_rest", "b2b", "opp_b2b", "three_in_four",
                 *[f"opp_{r}" for r in OPPONENT_RATES],
                 "n_fresh_absent", "miss_usage_share", *[f"miss_{s}_share" for s in STATS],
                 "net_gap", "abs_net_gap", "arena_team_id", "ot_game", "margin"]
    u = pl.merge(tg[team_cols].rename(columns={"GAME_DATE": "TEAM_GAME_DATE"}),
                 on=["GAME_ID", "TEAM_ID"], how="left", validate="many_to_one")
    if (u["GAME_DATE"] != u["TEAM_GAME_DATE"]).any():
        raise ValueError("player and team logs disagree on a game's date")

    u["season_phase"] = u["GAME_DATE"].dt.month.map(config.SEASON_PHASE_BY_MONTH)

    # Eligibility (§5.1), plus the neutral-site exclusion (Juan, 2026-10-17).
    u["eligible"] = ((u["n_prior"] >= config.MIN_PRIOR_GAMES)
                     & (u["base_min"] >= config.MIN_BASE_MINUTES)
                     & ~u["neutral_site"])
    for s in STATS:
        u[f"elig_{s}"] = u["eligible"] & (u[f"base_{s}"] >= config.STAT_FLOORS[s])

    # Outcomes (§5.2).
    u["min_resid"] = u["MIN"] - u["base_min"]
    for s in STATS:
        u[f"z_{s}"] = (u[s] - u[f"base_{s}"]) / u[f"sd_{s}"]
        u[f"hit_{s}"] = (u[s] > u[f"base_{s}"]).astype(float).where(u[f"base_{s}"].notna())
        u[f"rate_resid_{s}"] = (u[s] / u["MIN"] - u[f"base_rate_{s}"]).where(
            u["MIN"] >= config.RATE_RESID_MIN_MINUTES)
    u["early_exit"] = (u["MIN"] < config.EARLY_EXIT_FRAC * u["base_min"]).astype(float) \
        .where(u["base_min"].notna())

    u["SEASON"] = season
    return u[OUTPUT_COLUMNS].sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"],
                                         ignore_index=True)


def features_path(season: str):
    return config.FEATURES_DIR / f"{season}.parquet"


def build_season(season: str, *, after_lock: bool = False) -> pd.DataFrame:
    """Load a season's raw pulls, re-run the step-1 checks, and build its features.

    The confirm season is sealed until the lock (§2, §7). Only analyze.py
    --phase confirm passes after_lock=True, once its own lock checks pass."""
    if season == config.SEASON_CONFIRM and not after_lock:
        raise PermissionError(f"{season} is the confirm season; it is built only after the lock")
    player, team = pull.load_raw(season)
    pull.assert_sane(player, team, season)
    return build_features(player, team, season)


def main() -> int:
    season = config.SEASON_EXPLORE
    units = build_season(season)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    units.to_parquet(features_path(season), index=False)

    elig = units[units["eligible"]]
    print(f"{config.STUDY_VERSION} features, {season}: {len(units)} player-games with MIN > 0")
    print(f"  eligible (>= {config.MIN_PRIOR_GAMES} prior games, base_min >= "
          f"{config.MIN_BASE_MINUTES:g}, not neutral site): {len(elig)}")
    for s in STATS:
        print(f"  eligible for {s} (base_{s} >= {config.STAT_FLOORS[s]:g}): {int(units[f'elig_{s}'].sum())}")
    missing = elig[FEATURE_COLUMNS].isna().sum()
    print("  pre-game features missing on eligible rows: "
          + (", ".join(f"{c}={v}" for c, v in missing[missing > 0].items()) or "none"))
    print(f"  wrote {features_path(season).relative_to(config.STUDY_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

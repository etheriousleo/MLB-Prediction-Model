"""
A small synthetic league for the feature tests, shaped like pull.load_raw
output. Four teams play every game date (two games a date), so every team
has a game whenever another does. The date gaps mix back-to-backs,
three-in-fours and long rests. Each team has 8 players. The first 5 play
starter minutes (26-37), which makes them regulars once they have 10 games;
the other 3 play bench minutes (8-19), which never does.
"""
import numpy as np
import pandas as pd

TEAMS = (1, 2, 3, 4)
ABBR = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD"}
ROSTER = 8
STARTERS = 5
START = pd.Timestamp("2024-10-22")
GAPS = (1, 2, 1, 1, 3, 2, 1, 4)          # days between consecutive game dates
PAIRINGS = (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3)))
BOX = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
       "AST", "STL", "BLK", "TOV", "PF", "PTS"]


def game_dates(n_dates: int):
    dates = [START]
    for i in range(1, n_dates):
        dates.append(dates[-1] + pd.Timedelta(days=GAPS[i % len(GAPS)]))
    return dates


def schedule(n_dates: int):
    """(GAME_ID, date, home, away) for every game."""
    games, gid = [], 0
    for i, date in enumerate(game_dates(n_dates)):
        for a, b in PAIRINGS[i % len(PAIRINGS)]:
            home, away = (a, b) if (i // len(PAIRINGS)) % 2 == 0 else (b, a)
            gid += 1
            games.append((f"00224{gid:05d}", date, home, away))
    return games


def player_line(rng, minutes: int, playmaker: bool) -> dict:
    fga = rng.poisson(minutes * 0.42)
    fgm = rng.binomial(fga, 0.47)
    fg3a = rng.binomial(fga, 0.38)
    fg3m = min(rng.binomial(fg3a, 0.36), fgm)
    fta = rng.poisson(minutes * 0.10)
    ftm = rng.binomial(fta, 0.78)
    oreb, dreb = rng.poisson(minutes * 0.04), rng.poisson(minutes * 0.13)
    return dict(FGM=fgm, FGA=fga, FG3M=fg3m, FG3A=fg3a, FTM=ftm, FTA=fta, OREB=oreb,
                DREB=dreb, REB=oreb + dreb, AST=rng.poisson(minutes * (0.2 if playmaker else 0.07)),
                STL=rng.poisson(1), BLK=rng.poisson(0.5), TOV=rng.poisson(minutes * 0.05),
                PF=rng.poisson(2), PTS=2 * fgm + fg3m + ftm)


def make_season(n_dates: int = 34, seed: int = 0):
    """(player, team) frames. Team rows are the sums of their player rows."""
    rng = np.random.default_rng(seed)
    player_rows, team_rows = [], []
    for gid, date, home, away in schedule(n_dates):
        for tid, opp, matchup in ((home, away, f"{ABBR[home]} vs. {ABBR[away]}"),
                                  (away, home, f"{ABBR[away]} @ {ABBR[home]}")):
            lines = []
            for i in range(ROSTER):
                minutes = int(rng.integers(26, 38) if i < STARTERS else rng.integers(8, 20))
                line = player_line(rng, minutes, playmaker=(i == 0))
                lines.append(line)
                player_rows.append(dict(PLAYER_ID=tid * 100 + i, TEAM_ID=tid, GAME_ID=gid,
                                        GAME_DATE=date, MATCHUP=matchup, MIN=float(minutes), **line))
            team_rows.append(dict(TEAM_ID=tid, GAME_ID=gid, GAME_DATE=date, MATCHUP=matchup,
                                  MIN=240.0, **{c: sum(l[c] for l in lines) for c in BOX}))
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)


def team_game_ids(team: pd.DataFrame, team_id: int) -> list:
    """A team's GAME_IDs in date order; index k is its (k+1)-th game."""
    rows = team[team["TEAM_ID"] == team_id].sort_values("GAME_DATE")
    return rows["GAME_ID"].tolist()


def sit(player: pd.DataFrame, player_id: int, game_ids) -> pd.DataFrame:
    """The player doesn't appear in these games (DNPs aren't listed)."""
    return player[~((player["PLAYER_ID"] == player_id) & player["GAME_ID"].isin(list(game_ids)))]


def trade(player: pd.DataFrame, team: pd.DataFrame, player_id: int, new_team: int,
          from_date) -> pd.DataFrame:
    """From from_date on, the player suits up for new_team. Every team plays
    every date here, so the new team always has a game on his dates."""
    game_of = team.set_index(["TEAM_ID", "GAME_DATE"])["GAME_ID"]
    out = player.copy()
    moved = (out["PLAYER_ID"] == player_id) & (out["GAME_DATE"] >= from_date)
    out.loc[moved, "TEAM_ID"] = new_team
    out.loc[moved, "GAME_ID"] = [game_of[(new_team, d)] for d in out.loc[moved, "GAME_DATE"]]
    return out

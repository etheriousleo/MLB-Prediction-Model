"""
Spec §9 `test_pull_sanity`, plus tests that the checker and the pull logic do
what they claim.

- test_pull_sanity runs the §4 assertions on the committed raw data. It skips
  until pull.py has been run on a machine that can reach stats.nba.com.
- The synthetic tests build a 4-team league and corrupt it one way at a time,
  so each assertion is shown to catch the failure it exists for.
- The pull tests swap the API for a fake, so retries, pauses and skip-if-exists
  are covered without the network.
"""
import json

import numpy as np
import pandas as pd
import pytest

import config
import pull

# --- synthetic league --------------------------------------------------------

N_TEAMS, PLAYERS_PER_TEAM = 4, 10
GAMES_PER_TEAM = 2 * (N_TEAMS - 1)          # double round robin
SMALL = dict(n_teams=N_TEAMS, games_per_team=GAMES_PER_TEAM,
             expected_games=N_TEAMS * GAMES_PER_TEAM // 2)
ABBR = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD"}


def make_league(seed: int = 0):
    """(player, team) frames shaped like load_raw output: every team plays every
    other team once at home and once away, ten players a side, and player PTS
    sum to team PTS."""
    rng = np.random.default_rng(seed)
    pairs = [(h, a) for h in ABBR for a in ABBR if h != a]
    player_rows, team_rows = [], []
    for n, (home, away) in enumerate(pairs, start=1):
        gid = f"00224{n:05d}"
        date = pd.Timestamp("2024-10-22") + pd.Timedelta(days=n)
        for tid, opp, matchup in ((home, away, f"{ABBR[home]} vs. {ABBR[away]}"),
                                  (away, home, f"{ABBR[away]} @ {ABBR[home]}")):
            pts = rng.integers(0, 25, PLAYERS_PER_TEAM)
            for i in range(PLAYERS_PER_TEAM):
                player_rows.append(dict(
                    PLAYER_ID=tid * 100 + i, TEAM_ID=tid, GAME_ID=gid, GAME_DATE=date,
                    MATCHUP=matchup, MIN=float(rng.integers(5, 40)), FGA=10, FTA=2, OREB=1,
                    DREB=3, REB=4, AST=2, STL=1, BLK=0, TOV=1, PF=2, PTS=int(pts[i])))
            team_rows.append(dict(
                TEAM_ID=tid, GAME_ID=gid, GAME_DATE=date, MATCHUP=matchup, MIN=240.0,
                FGM=40, FGA=88, FG3M=12, FG3A=35, FTA=20, OREB=10, DREB=33, REB=43,
                AST=25, TOV=13, PTS=int(pts.sum())))
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)


def failed_checks(player, team):
    return {name for name, _ in pull.check_season(player, team, "synthetic", **SMALL).failures}


def test_synthetic_league_passes():
    player, team = make_league()
    pull.assert_sane(player, team, "synthetic", **SMALL)


def test_cup_final_is_caught():
    # The Cup final: GAME_ID prefix 006, an 83rd game for both finalists.
    player, team = make_league()
    cup_team = team[team["GAME_ID"] == team["GAME_ID"].iloc[0]].assign(GAME_ID="0062400001")
    cup_players = player[player["GAME_ID"] == player["GAME_ID"].iloc[0]].assign(GAME_ID="0062400001")
    team, player = pd.concat([team, cup_team]), pd.concat([player, cup_players])
    assert {"NBA Cup final absent", "unique GAME_IDs", "teams x games"} <= failed_checks(player, team)


def test_points_mismatch_is_caught_without_printing_points():
    player, team = make_league()
    team.loc[0, "PTS"] = 999
    res = pull.check_season(player, team, "synthetic", **SMALL)
    assert [name for name, _ in res.failures] == ["player PTS sum = team PTS"]
    detail = res.failures[0][1]
    assert team.loc[0, "GAME_ID"] in detail and "999" not in detail   # IDs, never values


def test_home_away_structure_is_caught():
    player, team = make_league()
    team.loc[1, "MATCHUP"] = team.loc[1, "MATCHUP"].replace("@", "vs.")   # two home rows
    assert failed_checks(player, team) == {"one home + one away row per game"}


def test_missing_team_row_is_caught():
    player, team = make_league()
    assert {"one home + one away row per game", "teams x games",
            "player and team logs cover the same team-games"} <= failed_checks(player, team.iloc[1:])


def test_duplicate_player_row_is_caught():
    player, team = make_league()
    player = pd.concat([player, player.iloc[[0]]])
    assert "no duplicate rows" in failed_checks(player, team)


def test_assert_sane_lists_every_failure():
    player, team = make_league()
    team.loc[0, "PTS"] = 999
    team.loc[1, "MATCHUP"] = team.loc[1, "MATCHUP"].replace("@", "vs.")
    with pytest.raises(pull.SanityError) as err:
        pull.assert_sane(player, team, "synthetic", **SMALL)
    assert "failed 2 check(s)" in str(err.value)


# --- non-playing players -----------------------------------------------------

def test_dnp_reading_when_dnps_are_absent():
    player, _ = make_league()
    ev = pull.dnp_evidence(player)
    assert ev["min_null_rows"] == 0 and ev["min_zero_rows"] == 0
    assert pull.dnp_reading(ev).startswith("DNPs are NOT listed")


def test_dnp_reading_when_dnps_are_listed():
    # Four DNP-CD rows per team-game: MIN 0 and a blank line -> 14 rows each.
    player, _ = make_league()
    dnps = []
    for (gid, tid), grp in player.groupby(["GAME_ID", "TEAM_ID"]):
        for j in range(4):
            row = grp.iloc[0].copy()
            row["PLAYER_ID"] = tid * 100 + 50 + j
            row["MIN"] = 0.0
            for col in pull._BOX_COLUMNS + ("REB",):
                row[col] = 0
            dnps.append(row)
    ev = pull.dnp_evidence(pd.concat([player, pd.DataFrame(dnps)]))
    assert ev["min_zero_blank_rows"] == 4 * ev["team_games"]
    assert pull.dnp_reading(ev).startswith("DNPs ARE listed")


def test_parse_minutes():
    assert pull.parse_minutes(pd.Series([34, 0, 12])).tolist() == [34.0, 0.0, 12.0]
    parsed = pull.parse_minutes(pd.Series(["34:30", "0:00", None, "", "12"], dtype=object))
    assert parsed.iloc[0] == 34.5 and parsed.iloc[1] == 0.0 and parsed.iloc[4] == 12.0
    assert parsed.iloc[2:4].isna().all()


# --- pull logic, with a fake API ---------------------------------------------

class FakeAPI:
    """Stands in for LeagueGameLog: fails the first `fail_first` calls."""

    def __init__(self, fail_first: int = 0):
        self.calls, self.fail_first = [], fail_first
        self.player, self.team = make_league()
        # The API sends GAME_DATE as a string and GAME_ID as a zero-padded string.
        for df in (self.player, self.team):
            df["GAME_DATE"] = df["GAME_DATE"].dt.strftime("%Y-%m-%d")

    def __call__(self, season, code):
        self.calls.append((season, code))
        if len(self.calls) <= self.fail_first:
            raise ConnectionError("simulated reset")
        return (self.player if code == "P" else self.team).copy()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "raw" / "manifest.json")
    return tmp_path


def test_fetch_retries_with_backoff(sandbox):
    api, waits = FakeAPI(fail_first=2), []
    df = pull.fetch_log("2024-25", "player", call=api, sleep=waits.append)
    assert len(df) == len(api.player)
    assert waits == list(config.API_BACKOFF_SECONDS[:2]) and len(api.calls) == 3


def test_fetch_gives_up_after_the_last_retry(sandbox):
    api, waits = FakeAPI(fail_first=99), []
    with pytest.raises(RuntimeError, match="4 attempts failed"):
        pull.fetch_log("2024-25", "team", call=api, sleep=waits.append)
    assert len(api.calls) == config.API_RETRIES + 1
    assert waits == list(config.API_BACKOFF_SECONDS)


def test_fetch_rejects_a_response_missing_columns(sandbox):
    def call(season, code):
        return FakeAPI().player.drop(columns=["MIN"])
    with pytest.raises(RuntimeError, match="missing columns"):
        pull.fetch_log("2024-25", "player", call=call, sleep=lambda s: None)


def test_pull_all_pauses_skips_existing_and_refreshes(sandbox):
    api, waits = FakeAPI(), []
    written = pull.pull_all(call=api, sleep=waits.append)
    assert len(written) == len(config.SEASONS) * len(config.LOG_KINDS) == len(api.calls)
    assert waits == [config.API_PAUSE_SECONDS] * (len(written) - 1)   # between calls only

    manifest = json.loads(config.MANIFEST_PATH.read_text())
    for path in written:
        assert manifest["files"][path.name]["sha256"] == pull.file_sha256(path)

    # Round-trip: what load_raw reads back passes the checks.
    player, team = pull.load_raw(config.SEASON_EXPLORE)
    assert player["GAME_ID"].str.startswith("002").all()
    pull.assert_sane(player, team, "synthetic", **SMALL)

    assert pull.pull_all(call=api, sleep=waits.append) == []          # all present: no calls
    assert len(api.calls) == len(written)
    assert len(pull.pull_all(refresh=True, call=api, sleep=waits.append)) == len(written)


# --- §9 test_pull_sanity: the committed raw data ------------------------------

@pytest.mark.parametrize("season", config.SEASONS)
def test_pull_sanity(season):
    if not all(pull.raw_path(season, kind).exists() for kind in config.LOG_KINDS):
        pytest.skip(f"{season} not pulled yet: run pull.py locally and commit data/raw/")
    player, team = pull.load_raw(season)
    pull.assert_sane(player, team, season)


def test_raw_files_match_manifest():
    files = sorted(config.RAW_DIR.glob("*.parquet")) if config.RAW_DIR.exists() else []
    if not files:
        pytest.skip("no raw data committed yet")
    manifest = json.loads(config.MANIFEST_PATH.read_text())["files"]
    for path in files:
        assert path.name in manifest, f"{path.name} is not in manifest.json"
        assert pull.file_sha256(path) == manifest[path.name]["sha256"], \
            f"{path.name} differs from what pull.py wrote"

"""
Spec §9 tests for features.py, plus hand checks of the team rates, rest
flags and eligibility.

- test_baseline_excludes_current: one player's baseline worked out by hand.
- test_future_perturbation: randomize every stat after a cutoff date D, and
  nothing dated on or before D may move. It runs on the synthetic league and
  on the real 2024-25 logs (the explore season only; 2025-26 stays sealed).
  A stricter variant also randomizes D's own stats and requires every
  pre-game column on D to hold. Tonight's participation is kept fixed in
  that variant, because fresh absences use it by design (§5.3).
- test_stint_reset: a traded player's baseline restarts with his new team.
- test_fresh_absence: one regular sits once and another sits 15 straight.

The expected values are computed with plain Python over the raw rows, not
with the pandas machinery under test.
"""
import statistics

import numpy as np
import pandas as pd
import pytest

import config
import features
import pull
import synthetic_league as sl

SEASON = "synthetic"


def build(player, team):
    return features.build_features(player, team, SEASON)


def player_rows(units, player_id):
    return units[units["PLAYER_ID"] == player_id].sort_values("GAME_DATE").reset_index(drop=True)


def team_feature(units, team_id, column):
    """A team-level feature per GAME_ID, read off the team's player rows."""
    return units[units["TEAM_ID"] == team_id].groupby("GAME_ID")[column].first()


# --- §9 test_baseline_excludes_current ---------------------------------------

def test_baseline_excludes_current():
    player, team = sl.make_season()
    pid = 101
    rows = player.index[player["PLAYER_ID"] == pid]          # already in date order
    pts = [12, 18, 9, 22, 15, 30, 11, 14, 20, 16, 25, 13, 40]
    mins = [30, 34, 28, 31, 36, 33, 27, 30, 35, 29, 32, 30, 38]
    player.loc[rows[:13], "PTS"] = pts
    player.loc[rows[:13], "MIN"] = [float(m) for m in mins]

    r = player_rows(build(player, team), pid)
    assert r.loc[0, "n_prior"] == 0 and np.isnan(r.loc[0, "base_PTS"])

    # The 13th game, by hand over the 12 before it.
    g = r.loc[12]
    base = statistics.mean(pts[:12])
    sd = max(statistics.stdev(pts[:12]), config.SD_FLOOR)
    base_rate = sum(pts[:12]) / sum(mins[:12])
    assert g["n_prior"] == 12
    assert g["base_PTS"] == pytest.approx(base)
    assert g["sd_PTS"] == pytest.approx(sd)
    assert g["base_min"] == pytest.approx(statistics.mean(mins[:12]))
    assert g["base_rate_PTS"] == pytest.approx(base_rate)
    assert g["z_PTS"] == pytest.approx((40 - base) / sd)
    assert g["hit_PTS"] == 1.0
    assert g["min_resid"] == pytest.approx(38 - statistics.mean(mins[:12]))
    assert g["rate_resid_PTS"] == pytest.approx(40 / 38 - base_rate)
    assert g["last5_dev_PTS"] == pytest.approx((statistics.mean(pts[7:12]) - base) / sd)
    assert g["last5_min_dev"] == pytest.approx(statistics.mean(mins[7:12]) - statistics.mean(mins[:12]))

    # Changing tonight's points moves tonight's outcome, never tonight's baseline.
    player.loc[rows[12], "PTS"] = 0
    r2 = player_rows(build(player, team), pid)
    assert r2.loc[12, "base_PTS"] == r.loc[12, "base_PTS"]
    assert r2.loc[12, "sd_PTS"] == r.loc[12, "sd_PTS"]
    assert r2.loc[12, "z_PTS"] < 0 < r.loc[12, "z_PTS"]


def test_sd_floor():
    player, team = sl.make_season()
    rows = player.index[player["PLAYER_ID"] == 101]
    player.loc[rows, "PTS"] = 10                               # zero spread
    r = player_rows(build(player, team), 101)
    assert (r.loc[2:, "sd_PTS"] == config.SD_FLOOR).all()
    assert np.isnan(r.loc[1, "sd_PTS"])                        # one prior game: no SD


# --- §9 test_future_perturbation ---------------------------------------------

TEAM_RANGES = {"FGM": (25, 50), "FGA": (70, 100), "FG3M": (5, 20), "FG3A": (20, 50),
               "FTM": (8, 30), "FTA": (10, 35), "OREB": (5, 16), "DREB": (25, 41),
               "AST": (15, 36), "STL": (3, 13), "BLK": (1, 9), "TOV": (8, 21),
               "PF": (12, 26), "PTS": (85, 136)}


def randomize(player, team, cutoff, *, include_cutoff=False, keep_participation=False, seed=7):
    """Every stat on dates after the cutoff replaced at random, in both logs.
    keep_participation leaves who-played intact: MIN > 0 stays > 0 and MIN 0 stays 0."""
    rng = np.random.default_rng(seed)
    p, t = player.copy(), team.copy()
    pm = (p["GAME_DATE"] >= cutoff) if include_cutoff else (p["GAME_DATE"] > cutoff)
    tm = (t["GAME_DATE"] >= cutoff) if include_cutoff else (t["GAME_DATE"] > cutoff)
    for col in sl.BOX:
        if col in p.columns:
            p.loc[pm, col] = rng.integers(0, 30, pm.sum())
    if keep_participation:
        played = pm & (p["MIN"] > 0)
        p.loc[played, "MIN"] = rng.integers(1, 49, played.sum()).astype(float)
    else:
        p.loc[pm, "MIN"] = rng.integers(0, 49, pm.sum()).astype(float)   # some become MIN 0
    for col, (lo, hi) in TEAM_RANGES.items():
        if col in t.columns:
            t.loc[tm, col] = rng.integers(lo, hi, tm.sum())
    t.loc[tm, "REB"] = t.loc[tm, "OREB"] + t.loc[tm, "DREB"]
    t.loc[tm, "MIN"] = rng.choice([240.0, 265.0], tm.sum())
    return p, t


def on_or_before(units, cutoff, columns):
    keep = units[units["GAME_DATE"] <= cutoff]
    return keep.set_index(["GAME_ID", "PLAYER_ID"])[columns].sort_index()


def check_no_look_ahead(player, team, season, cutoff, baseline=None):
    base = features.build_features(player, team, season) if baseline is None else baseline
    every = [c for c in features.OUTPUT_COLUMNS if c not in ("GAME_ID", "PLAYER_ID")]

    # The spec's test: stats after D randomized; nothing on or before D moves.
    p, t = randomize(player, team, cutoff)
    moved = features.build_features(p, t, season)
    pd.testing.assert_frame_equal(on_or_before(base, cutoff, every),
                                  on_or_before(moved, cutoff, every), check_exact=True)
    after = lambda u: u.loc[u["GAME_DATE"] > cutoff, "base_PTS"].dropna()   # noqa: E731
    assert not after(base).equals(after(moved)), "randomizing changed nothing; the test is vacuous"

    # Stricter: D's own stats randomized too; every pre-game column on D holds.
    p, t = randomize(player, team, cutoff, include_cutoff=True, keep_participation=True)
    same_day = features.build_features(p, t, season)
    pd.testing.assert_frame_equal(on_or_before(base, cutoff, features.PREGAME_COLUMNS),
                                  on_or_before(same_day, cutoff, features.PREGAME_COLUMNS),
                                  check_exact=True)


def test_future_perturbation_synthetic():
    player, team = sl.make_season()
    # A trade and absences before the cutoff, so stints and teammates are exercised.
    dates = sl.game_dates(34)
    player = sl.trade(player, team, 102, new_team=3, from_date=dates[8])
    player = sl.sit(player, 100, sl.team_game_ids(team, 1)[12:14])
    check_no_look_ahead(player, team, SEASON, dates[20])


@pytest.fixture(scope="module")
def explore_raw():
    season = config.SEASON_EXPLORE
    if not all(pull.raw_path(season, k).exists() for k in config.LOG_KINDS):
        pytest.skip(f"{season} not pulled yet")
    return pull.load_raw(season)


@pytest.fixture(scope="module")
def explore_units(explore_raw):
    return features.build_features(*explore_raw, config.SEASON_EXPLORE)


def test_future_perturbation_explore_season(explore_raw, explore_units):
    player, team = explore_raw
    # The first game date after the All-Star break: past the trade deadline,
    # so the "before" side has every kind of stint and absence in it.
    cutoff = team.loc[team["GAME_DATE"] >= "2025-02-20", "GAME_DATE"].min()
    check_no_look_ahead(player, team, config.SEASON_EXPLORE, cutoff, baseline=explore_units)


# --- §9 test_stint_reset -------------------------------------------------------

def test_stint_reset():
    player, team = sl.make_season()
    trade_date = sl.game_dates(34)[15]
    player = sl.trade(player, team, 102, new_team=3, from_date=trade_date)
    r = player_rows(build(player, team), 102)
    before, after = r[r["GAME_DATE"] < trade_date], r[r["GAME_DATE"] >= trade_date].reset_index(drop=True)

    assert (before["TEAM_ID"] == 1).all() and (before["stint"] == 1).all()
    assert (after["TEAM_ID"] == 3).all() and (after["stint"] == 2).all()
    assert before["n_prior"].tolist() == list(range(len(before)))
    first = after.loc[0]
    assert first["n_prior"] == 0 and np.isnan(first["base_PTS"]) and np.isnan(first["base_min"])
    assert not first["eligible"]
    eleventh = after.loc[10]
    assert eleventh["n_prior"] == 10
    assert eleventh["base_PTS"] == pytest.approx(statistics.mean(after.loc[:9, "PTS"]))
    assert eleventh["base_min"] == pytest.approx(statistics.mean(after.loc[:9, "MIN"]))


# --- §9 test_fresh_absence ----------------------------------------------------

def mean_usage(df):
    return statistics.mean(df["FGA"] + config.POSS_FTA_COEF * df["FTA"] + df["TOV"])


def test_fresh_absence():
    player, team = sl.make_season()
    g = sl.team_game_ids(team, 1)                   # g[k] is team 1's game k+1
    full = player.copy()
    player = sl.sit(player, 101, g[11:26])          # a starter sits games 12-26: 15 straight
    player = sl.sit(player, 100, [g[21]])           # another sits game 22 only
    player = sl.sit(player, 105, [g[19]])           # a bench player (never a regular) sits game 20
    units = build(player, team)
    counts = team_feature(units, 1, "n_fresh_absent").reindex(g).tolist()

    expected = [0] * len(g)
    for k in range(11, 16):     # games 12-16: 101 played in one of the previous five
        expected[k] = 1
    expected[21] = 1            # game 22: only the one-game sitter; 101 is out too, but long out
    assert counts == expected

    # Shares by hand. Game 22: 100's average over games 1-21, over the team's.
    share = team_feature(units, 1, "miss_usage_share")
    p100 = full[(full["PLAYER_ID"] == 100) & full["GAME_ID"].isin(g[:21])]
    t1 = team[(team["TEAM_ID"] == 1) & team["GAME_ID"].isin(g[:21])]
    assert share[g[21]] == pytest.approx(mean_usage(p100) / mean_usage(t1))
    pts_share = team_feature(units, 1, "miss_PTS_share")
    assert pts_share[g[21]] == pytest.approx(p100["PTS"].mean() / t1["PTS"].mean())
    # Game 16, the fifth game 101 misses: his averages stop at his last game (11),
    # and the team's run through game 15.
    p101 = full[(full["PLAYER_ID"] == 101) & full["GAME_ID"].isin(g[:11])]
    t1_15 = team[(team["TEAM_ID"] == 1) & team["GAME_ID"].isin(g[:15])]
    assert share[g[15]] == pytest.approx(mean_usage(p101) / mean_usage(t1_15))
    assert share[g[16]] == 0.0


def test_traded_regular_leaves_his_old_team():
    player, team = sl.make_season()
    dates = sl.game_dates(34)
    player = sl.trade(player, team, 103, new_team=2, from_date=dates[20])
    counts = team_feature(build(player, team), 1, "n_fresh_absent")
    g = sl.team_game_ids(team, 1)
    # Trade night: he's gone from team 1 and hasn't yet shown up elsewhere
    # before tonight, so his old team feels a fresh absence...
    assert counts[g[20]] == 1
    # ...and once he has played for team 2, he's no longer team 1's.
    assert counts[g[21]] == 0


# --- rest flags, team rates, eligibility, neutral sites ----------------------

def test_rest_flags():
    player, team = sl.make_season()
    units = build(player, team)
    dates = sl.game_dates(34)
    g = sl.team_game_ids(team, 1)                   # team 1 plays every date
    rest = team_feature(units, 1, "days_rest").reindex(g)
    b2b = team_feature(units, 1, "b2b").reindex(g)
    three = team_feature(units, 1, "three_in_four").reindex(g)
    opp_b2b = team_feature(units, 1, "opp_b2b").reindex(g)
    assert np.isnan(rest.iloc[0]) and np.isnan(b2b.iloc[0])
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        in_prior_3_days = sum(1 for d in dates[:i] if 1 <= (dates[i] - d).days <= 3)
        assert rest.iloc[i] == gap - 1
        assert b2b.iloc[i] == float(gap == 1)
        assert opp_b2b.iloc[i] == float(gap == 1)   # every opponent also played the day before
        assert three.iloc[i] == float(in_prior_3_days >= 2)


def team_sums_before(team, team_id, date):
    """Plain-Python season-to-date sums for one team, or the league (None)."""
    rows = team[team["GAME_DATE"] < date]
    by_game = {gid: grp for gid, grp in rows.groupby("GAME_ID")}
    s = dict(games=0, fgm=0.0, fga=0.0, allowed=0.0, poss=0.0, game_min=0.0)
    for gid, grp in by_game.items():
        for _, me in grp.iterrows():
            if team_id is not None and me["TEAM_ID"] != team_id:
                continue
            them = grp[grp["TEAM_ID"] != me["TEAM_ID"]].iloc[0]
            est = [r["FGA"] + 0.44 * r["FTA"] - r["OREB"] + r["TOV"] for r in (me, them)]
            s["games"] += 1
            s["fgm"] += me["FGM"]
            s["fga"] += me["FGA"]
            s["allowed"] += them["PTS"]
            s["poss"] += sum(est) / 2
            s["game_min"] += me["MIN"] / 5
    return s


def test_team_rates_by_hand():
    player, team = sl.make_season()
    units = build(player, team)
    row = units[(units["TEAM_ID"] == 1) & (units["GAME_ID"] == sl.team_game_ids(team, 1)[12])].iloc[0]
    opp, date, k = row["OPP_TEAM_ID"], row["GAME_DATE"], config.SHRINK_K
    o, lg = team_sums_before(team, opp, date), team_sums_before(team, None, date)
    n = o["games"]
    shrink = lambda mine, league: (n * mine + k * league) / (n + k)   # noqa: E731
    assert row["opp_fg_pct"] == pytest.approx(shrink(o["fgm"] / o["fga"], lg["fgm"] / lg["fga"]))
    assert row["opp_drtg"] == pytest.approx(
        shrink(100 * o["allowed"] / o["poss"], 100 * lg["allowed"] / lg["poss"]))
    assert row["opp_pace"] == pytest.approx(
        shrink(48 * o["poss"] / o["game_min"], 48 * lg["poss"] / lg["game_min"]))


def test_eligibility():
    player, team = sl.make_season()
    units = build(player, team)
    starter, bench = player_rows(units, 100), player_rows(units, 106)
    assert not starter.loc[:9, "eligible"].any()          # fewer than 10 prior games
    assert starter.loc[10:, "eligible"].all()             # starter minutes from game 11 on
    assert not bench["eligible"].any()                    # base_min under 20
    for s in config.STATS:
        floor = config.STAT_FLOORS[s]
        assert (units[f"elig_{s}"] == (units["eligible"] & (units[f"base_{s}"] >= floor))).all()


def test_neutral_site_counts_backward_but_is_never_a_unit(monkeypatch):
    player, team = sl.make_season()
    g = sl.team_game_ids(team, 1)
    neutral = g[14]
    is_it = team["GAME_ID"] == neutral
    team.loc[is_it, "MATCHUP"] = team.loc[is_it, "MATCHUP"].str.replace("vs.", "@", regex=False)
    monkeypatch.setattr(config, "NEUTRAL_SITE_GAMES", {SEASON: {neutral: "Paris"}})
    units = build(player, team)

    rows = units[units["GAME_ID"] == neutral]
    assert rows["neutral_site"].all() and not rows["eligible"].any()
    assert rows["arena_team_id"].isna().all()
    nxt = units[(units["PLAYER_ID"] == 100) & (units["GAME_ID"] == g[15])].iloc[0]
    assert nxt["n_prior"] == 15 and nxt["eligible"]       # the neutral game still counted
    assert nxt["arena_team_id"] in (1, nxt["OPP_TEAM_ID"])


# --- the real explore season ----------------------------------------------------

def test_explore_season_features(explore_raw, explore_units):
    player, _ = explore_raw
    u = explore_units
    assert len(u) == int((player["MIN"] > 0).sum())           # every MIN > 0 row is a unit
    assert not u.duplicated(["GAME_ID", "PLAYER_ID"]).any()
    neutral = u["GAME_ID"].isin(list(config.NEUTRAL_SITE_GAMES[config.SEASON_EXPLORE]))
    assert u.loc[neutral, "neutral_site"].all() and not u.loc[neutral, "eligible"].any()
    assert not u.loc[~neutral, "neutral_site"].any()
    elig = u[u["eligible"]]
    assert len(elig) > 10_000
    missing = elig[features.FEATURE_COLUMNS].isna().sum()
    assert missing.sum() == 0, f"eligible rows with missing features: {missing[missing > 0].to_dict()}"


def test_confirm_season_is_sealed():
    with pytest.raises(PermissionError):
        features.build_season(config.SEASON_CONFIRM)

"""
Step 1 of NBA_STUDY_SPEC.md (§4): pull the regular-season game logs and
sanity-check them.

Run this on a home machine. stats.nba.com refuses most cloud IPs; from the
Claude Code sandbox it returned 403 on 2026-10-13.

    python pull.py            # pull whatever is missing, then check everything
    python pull.py --refresh  # re-pull every file even if it exists

Writes:
  data/raw/{season}_{player|team}.parquet   raw LeagueGameLog output, committed
  data/raw/manifest.json                    when/how each file was pulled + sha256
  reports/pull_sanity.md                    row counts, check results, DNP finding

Outcome-free by design. Failure messages name IDs, never stat values, and
the confirm season (2025-26) gets row counts and pass/fail only: nothing
about its outcomes is printed before the lock (§7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd

import config


class SanityError(AssertionError):
    """A §4 assertion failed. The message lists every failure for the season."""


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def raw_path(season: str, kind: str):
    return config.RAW_DIR / f"{season}_{kind}.parquet"


def _league_game_log(season: str, code: str) -> pd.DataFrame:
    """One LeagueGameLog call. nba_api is imported here so that tests and the
    analysis code never need it (or the network)."""
    from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog

    resp = LeagueGameLog(
        season=season,
        player_or_team_abbreviation=code,
        season_type_all_star=config.SEASON_TYPE,
        timeout=config.API_TIMEOUT,
    )
    return resp.get_data_frames()[0]


def fetch_log(season: str, kind: str, *, call=_league_game_log, sleep=time.sleep) -> pd.DataFrame:
    """Fetch one log with up to API_RETRIES retries and backoff.

    Any exception from the call is retried: timeouts, connection resets, and the
    JSON errors you get when stats.nba.com serves a block page instead of data.
    A response that arrives but is malformed (empty or missing columns) is
    not retried, because asking again won't fix a wrong season string.
    """
    code = config.LOG_KINDS[kind]
    for attempt in range(config.API_RETRIES + 1):
        try:
            df = call(season, code)
            break
        except Exception as exc:  # noqa: BLE001 - see docstring
            if attempt == config.API_RETRIES:
                raise RuntimeError(
                    f"{season} {kind}: {attempt + 1} attempts failed; last error: {exc!r}"
                ) from exc
            wait = config.API_BACKOFF_SECONDS[attempt]
            print(f"  {season} {kind}: attempt {attempt + 1} failed ({exc!r}); retrying in {wait}s")
            sleep(wait)

    if df.empty:
        raise RuntimeError(f"{season} {kind}: the API returned zero rows")
    missing = [c for c in config.REQUIRED_COLUMNS[kind] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{season} {kind}: response is missing columns {missing}")
    return df


def pull_all(refresh: bool = False, *, call=_league_game_log, sleep=time.sleep) -> list:
    """Pull every (season, kind) whose file is missing, or all of them with
    refresh=True. Pauses API_PAUSE_SECONDS between consecutive calls. Returns
    the paths written."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for season in config.SEASONS:
        for kind in config.LOG_KINDS:
            path = raw_path(season, kind)
            if path.exists() and not refresh:
                print(f"  skip {path.name} (exists; pass --refresh to re-pull)")
                continue
            if written:
                sleep(config.API_PAUSE_SECONDS)
            print(f"  pull {season} {kind} ...")
            df = fetch_log(season, kind, call=call, sleep=sleep)
            # GAME_ID is a zero-padded string ("0022400001"); keep it that way.
            df["GAME_ID"] = df["GAME_ID"].astype(str)
            # Write-then-rename, so an interrupted run never leaves a partial
            # file that the next run would skip as already pulled.
            tmp = path.with_name(path.name + ".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
            written.append(path)
    if written:
        _record_in_manifest(written)
    return written


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_stamp() -> str:
    # Timezone-aware UTC, never a naive local time (the CLAUDE.md date rule).
    # UTC rather than ET because this stamps a file, not a slate, and UTC needs
    # no tz database on Windows.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nba_api_version() -> str:
    try:
        return metadata.version("nba_api")
    except metadata.PackageNotFoundError:
        return "unknown"


def _record_in_manifest(paths) -> None:
    """Record the files just written. Entries for files not re-pulled are left
    alone, so a hand-edited raw file shows up as a hash mismatch in the tests."""
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8")) if config.MANIFEST_PATH.exists() else {}
    files = manifest.setdefault("files", {})
    for path in paths:
        files[path.name] = {
            "pulled_at_utc": _utc_stamp(),
            "nba_api": _nba_api_version(),
            "rows": int(len(pd.read_parquet(path, columns=["GAME_ID"]))),
            "sha256": file_sha256(path),
        }
    manifest["study_version"] = config.STUDY_VERSION
    manifest["season_type"] = config.SEASON_TYPE
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def parse_minutes(col: pd.Series) -> pd.Series:
    """MIN as float minutes. LeagueGameLog sends whole minutes; "MM:SS"
    strings (other endpoints' format) are handled in case that ever changes."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)

    def one(v):
        if pd.isna(v) or str(v).strip() == "":
            return np.nan
        text = str(v).strip()
        if ":" in text:
            mins, secs = text.split(":", 1)
            return float(mins) + float(secs) / 60.0
        return float(text)

    return col.map(one).astype(float)


def load_raw(season: str):
    """Read one season's committed pulls, typed for analysis. Returns (player, team)."""
    frames = []
    for kind in ("player", "team"):
        path = raw_path(season, kind)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run pull.py on a machine that can reach stats.nba.com."
            )
        df = pd.read_parquet(path)
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["MIN"] = parse_minutes(df["MIN"])
        frames.append(df)
    return frames[0], frames[1]


# ---------------------------------------------------------------------------
# §4 sanity assertions
# ---------------------------------------------------------------------------

@dataclass
class SeasonCheck:
    season: str
    facts: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)      # (name, passed, detail)

    @property
    def failures(self):
        return [(name, detail) for name, ok, detail in self.checks if not ok]

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, bool(passed), detail))


def _ids(values, limit: int = 10) -> str:
    values = list(values)
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f" ... (+{len(values) - limit} more)" if len(values) > limit else "")


def check_season(player: pd.DataFrame, team: pd.DataFrame, season: str = "", *,
                 n_teams: int = config.N_TEAMS,
                 games_per_team: int = config.GAMES_PER_TEAM,
                 expected_games: int = config.EXPECTED_GAMES,
                 game_id_prefix: str = config.REGULAR_SEASON_GAME_ID_PREFIX) -> SeasonCheck:
    """Run every §4 assertion on one season and collect the results (it does
    not raise; see assert_sane). Expects frames from load_raw. The keyword
    arguments exist only so the tests can use a small synthetic league."""
    res = SeasonCheck(season)
    games = team["GAME_ID"].unique()
    res.facts.update(player_rows=len(player), team_rows=len(team), unique_games=len(games))

    # Unique games. On a mismatch, the per-team counts and stray GAME_IDs below
    # say where the discrepancy is.
    res.add("unique GAME_IDs", len(games) == expected_games,
            f"{len(games)} found, {expected_games} expected")

    # 30 teams x 82 games. A team at 83 means a non-regular-season game got in.
    per_team = team.groupby("TEAM_ID").size()
    off = per_team[per_team != games_per_team]
    detail = f"{len(per_team)} teams (expected {n_teams}); "
    if off.empty:
        detail += f"every team at {games_per_team} games"
    else:
        detail += f"TEAM_IDs not at {games_per_team} games: " + _ids(f"{t}={n}" for t, n in off.items())
    res.add("teams x games", len(per_team) == n_teams and off.empty, detail)

    # NBA Cup championship game absent: regular-season GAME_IDs all start "002".
    stray = sorted(set(team["GAME_ID"]).union(player["GAME_ID"]))
    stray = [g for g in stray if not g.startswith(game_id_prefix)]
    res.add("NBA Cup final absent", not stray,
            "no non-regular-season GAME_IDs" if not stray
            else f"GAME_IDs without the {game_id_prefix} prefix: {_ids(stray)}")

    # Exactly two team rows per game: one home ("vs."), one away ("@").
    is_home = team["MATCHUP"].str.contains("vs.", regex=False)
    is_away = team["MATCHUP"].str.contains("@", regex=False)
    per_game = pd.DataFrame({"GAME_ID": team["GAME_ID"], "home": is_home, "away": is_away}) \
        .groupby("GAME_ID").agg(rows=("home", "size"), home=("home", "sum"), away=("away", "sum"))
    bad = per_game[(per_game["rows"] != 2) | (per_game["home"] != 1) | (per_game["away"] != 1)]
    res.add("one home + one away row per game", bad.empty,
            "all games" if bad.empty else f"{len(bad)} games off: {_ids(bad.index)}")

    # No duplicate rows.
    dup_team = team.duplicated(["GAME_ID", "TEAM_ID"]).sum()
    dup_player = player.duplicated(["GAME_ID", "PLAYER_ID"]).sum()
    res.add("no duplicate rows", dup_team == 0 and dup_player == 0,
            f"{dup_team} duplicate team-games, {dup_player} duplicate player-games")

    # Both logs describe the same team-games (every join is by ID).
    team_keys = set(zip(team["GAME_ID"], team["TEAM_ID"]))
    player_keys = set(zip(player["GAME_ID"], player["TEAM_ID"]))
    only_team, only_player = team_keys - player_keys, player_keys - team_keys
    res.add("player and team logs cover the same team-games", not only_team and not only_player,
            f"{len(only_team)} team-games without player rows, "
            f"{len(only_player)} player team-games missing from the team log"
            + (f": {_ids(sorted(only_team | only_player))}" if only_team or only_player else ""))

    # Sum of player PTS equals team PTS for every team-game. IDs only: no points
    # are printed, so the check is safe to run on the confirm season.
    summed = player.groupby(["GAME_ID", "TEAM_ID"])["PTS"].sum()
    joined = team.set_index(["GAME_ID", "TEAM_ID"])["PTS"].to_frame("team").join(
        summed.rename("players"), how="left")
    mismatch = joined[joined["team"] != joined["players"]]
    res.add("player PTS sum = team PTS", mismatch.empty,
            "every team-game" if mismatch.empty
            else f"{len(mismatch)} team-games differ: {_ids(f'{g}/{t}' for g, t in mismatch.index)}")

    res.facts["dnp"] = dnp_evidence(player)
    return res


def assert_sane(player: pd.DataFrame, team: pd.DataFrame, season: str = "", **kwargs) -> SeasonCheck:
    """check_season, raising SanityError that lists every failure."""
    res = check_season(player, team, season, **kwargs)
    if res.failures:
        lines = "\n".join(f"  - {name}: {detail}" for name, detail in res.failures)
        raise SanityError(f"{season or 'season'} failed {len(res.failures)} check(s):\n{lines}")
    return res


# ---------------------------------------------------------------------------
# Non-playing players (§4 last bullet; feeds §5.3 fresh absences)
# ---------------------------------------------------------------------------

# Box-score columns that are all zero for a player who never got on the floor.
_BOX_COLUMNS = ("FGA", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "PTS")


def dnp_evidence(player: pd.DataFrame) -> dict:
    """Do players who did not play appear in the player log?

    A DNP listing would show up as MIN null, or MIN 0 with a blank line, at
    roughly 2-5 per team-game, and push the rows per team-game toward a full
    active list (13+). Players who did play show up as MIN > 0 rows. A MIN-0
    row with stats is a cameo of under 30 seconds, rounded down.
    """
    mins = player["MIN"]
    box = [c for c in _BOX_COLUMNS if c in player.columns]
    blank = player[box].fillna(0).eq(0).all(axis=1)
    keys = ["GAME_ID", "TEAM_ID"]
    rows_per = player.groupby(keys).size()
    played_per = player[mins > 0].groupby(keys).size()

    def spread(s):
        return (int(s.min()), float(s.median()), int(s.max())) if len(s) else (0, 0.0, 0)

    return {
        "team_games": int(len(rows_per)),
        "min_null_rows": int(mins.isna().sum()),
        "min_zero_rows": int(mins.eq(0).sum()),
        "min_zero_blank_rows": int((mins.eq(0) & blank).sum()),
        "min_zero_with_stats_rows": int((mins.eq(0) & ~blank).sum()),
        "rows_per_team_game": spread(rows_per),
        "played_rows_per_team_game": spread(played_per),
    }


def dnp_reading(ev: dict) -> str:
    """A plain-language reading of dnp_evidence. Provisional: a human confirms it
    in the step-1 report. features.py does not depend on it either way."""
    dnp_like = ev["min_null_rows"] + ev["min_zero_blank_rows"]
    if dnp_like == 0:
        return "DNPs are NOT listed: a player who did not play has no row."
    if dnp_like < 0.05 * ev["team_games"]:
        return (f"DNPs are NOT listed: {dnp_like} MIN-null/MIN-0 blank rows is far too few for a "
                f"DNP listing (that would be ~2-5 per team-game, i.e. thousands); they look like "
                f"cameos of under 30 seconds.")
    if ev["rows_per_team_game"][1] >= 13:
        return (f"DNPs ARE listed: {dnp_like} MIN-null/MIN-0 blank rows, and a median of "
                f"{ev['rows_per_team_game'][1]:g} rows per team-game (a full active list).")
    return (f"UNCLEAR: {dnp_like} MIN-null/MIN-0 blank rows but a median of only "
            f"{ev['rows_per_team_game'][1]:g} rows per team-game. Inspect before building features.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(results) -> str:
    out = [
        f"# Pull sanity: {config.STUDY_VERSION}",
        "",
        f"Generated by `pull.py` at {_utc_stamp()} with nba_api {_nba_api_version()}. "
        "Outcome-free: row counts, pass/fail, and the non-playing-player finding.",
        "",
    ]
    for res in results:
        confirm = res.season == config.SEASON_CONFIRM
        role = "confirm: row counts and pass/fail only until the lock" if confirm else "explore"
        f = res.facts
        out += [
            f"## {res.season} ({role})",
            "",
            f"- Player rows: {f.get('player_rows', 'n/a')}",
            f"- Team rows: {f.get('team_rows', 'n/a')}",
            f"- Unique games: {f.get('unique_games', 'n/a')} (expected {config.EXPECTED_GAMES})",
            "",
            "| Check | Result | Detail |",
            "|---|---|---|",
        ]
        out += [f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |" for name, ok, detail in res.checks]
        out.append("")

        ev = f.get("dnp")
        if ev is None:
            continue
        out += ["### Non-playing players", ""]
        out += [
            f"- Rows with MIN null: {ev['min_null_rows']}",
            f"- Rows with MIN 0: {ev['min_zero_rows']} "
            f"(blank line: {ev['min_zero_blank_rows']}; with stats: {ev['min_zero_with_stats_rows']})",
        ]
        if not confirm:
            lo, med, hi = ev["rows_per_team_game"]
            plo, pmed, phi = ev["played_rows_per_team_game"]
            out += [
                f"- Player rows per team-game: min {lo}, median {med:g}, max {hi} "
                f"(over {ev['team_games']} team-games)",
                f"- Rows with MIN > 0 per team-game: min {plo}, median {pmed:g}, max {phi}",
                "",
                f"**Reading (provisional):** {dnp_reading(ev)}",
                "",
                "Either way, `features.py` counts a player as having played only when he has a "
                "row with MIN > 0, so fresh-absence detection (§5.3) works under both conventions.",
            ]
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pull the study's game logs and sanity-check them (spec §4).")
    parser.add_argument("--refresh", action="store_true", help="re-pull files that already exist")
    args = parser.parse_args(argv)

    print(f"{config.STUDY_VERSION}: pulling {', '.join(config.SEASONS)} ({config.SEASON_TYPE})")
    pull_all(refresh=args.refresh)

    results = []
    for season in config.SEASONS:
        player, team = load_raw(season)
        results.append(check_season(player, team, season))

    report = render_report(results)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # UTF-8 explicitly: the report has "§", and Windows would otherwise
    # write it in the locale code page.
    config.PULL_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)

    failed = [r for r in results if r.failures]
    if failed:
        print(f"\nSANITY FAILED for {', '.join(r.season for r in failed)}. Stop here and report "
              f"the discrepancy (see {config.PULL_REPORT_PATH.name}).", file=sys.stderr)
        return 1
    print(f"\nAll checks passed. Commit data/raw/ and {config.PULL_REPORT_PATH.relative_to(config.STUDY_DIR)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

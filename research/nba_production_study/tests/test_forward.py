"""
Tests for v2, the forward check on 2026-27 (spec §12). None of them read
2025-26 or 2026-27 data.

- The pull can't fetch 2026-27 before its regular season is over.
- 2026-27 is sealed in features, and --phase forward refuses before it
  reads anything.
- The v2 lock (the draft until it's committed) carries exactly the
  hypotheses v1 confirmed, with v1's model, rule, and 2024-25 numbers.
- The forward report shows 2024-25, 2025-26 and 2026-27 side by side (run
  on 2024-25 standing in for 2026-27).
"""
import re
from datetime import date

import pytest

import analyze
import config
import features
import pull
from test_pull_sanity import FakeAPI


def test_forward_season_waits_for_the_end_of_its_regular_season():
    open_on = date.fromisoformat(config.FORWARD_PULL_NOT_BEFORE)
    assert open_on > date(2027, 4, 11)                     # the 2026-27 regular season's last day
    assert not pull.forward_season_open(date(2027, 4, 12))
    assert pull.forward_season_open(open_on)
    assert pull.seasons_to_pull(date(2026, 10, 17)) == config.SEASONS
    assert pull.seasons_to_pull(open_on) == (*config.SEASONS, config.SEASON_FORWARD)


def test_pull_all_skips_the_forward_season_until_it_opens(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "raw" / "manifest.json")
    api = FakeAPI()
    pull.pull_all(call=api, sleep=lambda s: None, today=date(2027, 1, 15))
    assert {season for season, _ in api.calls} == set(config.SEASONS)
    pull.pull_all(call=api, sleep=lambda s: None,
                  today=date.fromisoformat(config.FORWARD_PULL_NOT_BEFORE))
    assert api.calls[-2:] == [(config.SEASON_FORWARD, "P"), (config.SEASON_FORWARD, "T")]


def test_forward_season_is_sealed():
    with pytest.raises(PermissionError):
        features.build_season(config.SEASON_FORWARD)


def test_forward_refuses_before_reading_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FORWARD_LOCK_PATH", tmp_path / "LOCKED_HYPOTHESES_V2.md")
    monkeypatch.setattr(config, "FORWARD_REPORT_PATH", tmp_path / "forward.md")

    def sealed(*args, **kwargs):
        raise AssertionError("2026-27 was read before the lock checks passed")
    monkeypatch.setattr(features, "build_season", sealed)
    monkeypatch.setattr(pull, "load_raw", sealed)
    with pytest.raises(analyze.LockError, match="doesn't exist"):
        analyze.forward()


def v2_lock():
    """The committed v2 lock if it exists, else the draft (read as if locked)."""
    for path in (config.FORWARD_LOCK_PATH, config.FORWARD_LOCK_DRAFT_PATH):
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return analyze.read_lock(text.replace('"status": "draft"', '"status": "locked"'))
    pytest.skip("no v2 lock or draft")


def test_v2_lock_is_v1s_confirmed_hypotheses():
    lock = v2_lock()
    analyze.check_lock_matches_code(lock, config.FORWARD_STUDY_VERSION)
    v1 = analyze.read_lock(config.LOCK_PATH.read_text(encoding="utf-8"))
    report = config.CONFIRM_REPORT_PATH.read_text(encoding="utf-8")
    confirmed = re.findall(r"^\| (H\d+) \|.*\| \*\*confirmed\*\* \|$", report, flags=re.M)
    assert [h["id"] for h in lock["hypotheses"]] == confirmed
    by_id = {h["id"]: h for h in v1["hypotheses"]}
    for h in lock["hypotheses"]:
        old = by_id[h["id"]]
        for key in ("stat", "feature", "direction", "explore_coef", "explore_ci"):
            assert h[key] == old[key], (h["id"], key)
    for key in ("model", "practical_floor_sd", "alpha_one_sided"):
        assert lock[key] == v1[key]


@pytest.fixture(scope="module")
def explore_units():
    season = config.SEASON_EXPLORE
    if not all(pull.raw_path(season, k).exists() for k in config.LOG_KINDS):
        pytest.skip(f"{season} not pulled yet")
    return features.build_season(season)


def test_forward_report_layout(explore_units):
    lock = v2_lock()
    r = analyze.evaluate_locked(explore_units, lock)
    text = analyze.write_locked_report(r, lock, "abc1234", analyze.provenance(), "forward")
    assert "v2's one look at 2026-27" in text and "`LOCKED_HYPOTHESES_V2.md` at commit `abc1234`" in text
    header = next(line for line in text.splitlines() if line.startswith("| ID |"))
    assert "2024-25 effect" in header and "2025-26 effect" in header and "2026-27 effect" in header
    assert f"of {len(lock['hypotheses'])} hypotheses confirmed" in text

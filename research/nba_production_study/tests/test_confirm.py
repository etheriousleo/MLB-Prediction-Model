"""
Tests for Phase B (spec §7-8), none of which read the confirm season.

- The confirmation rule on hand-made numbers: sign, Holm, and floor each
  fail on their own.
- Lock parsing: one JSON block, status "locked", and coefficients that
  exist in the locked model.
- Every §8 refusal, in a scratch git repo: no lock, an uncommitted lock, a
  modified lock, code changed after the lock, and an existing report.
- confirm() refuses before it reads anything when there's no lock.
- The full evaluation path, run on 2024-25 standing in for the confirm
  season: a real effect confirms, while the wrong direction and a null
  feature don't.
"""
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

import analyze
import config
import features
import pull


def lock_text(status="locked", hypotheses=None) -> str:
    lock = dict(
        study_version=config.STUDY_VERSION, status=status, approved_by="test",
        approved_on="2026-10-17", practical_floor_sd=config.PRACTICAL_FLOOR_SD,
        alpha_one_sided=config.CONFIRM_ALPHA,
        model=dict(common_drivers=list(config.COMMON_DRIVERS),
                   per_stat_drivers=list(config.PER_STAT_DRIVERS),
                   binary_drivers=list(config.BINARY_DRIVERS)),
        hypotheses=hypotheses if hypotheses is not None else [
            dict(id="H1", stat="PTS", feature="miss_PTS_share", direction="+",
                 explore_coef=0.15, explore_ci=[0.12, 0.18])])
    return f"# Lock\n\nprose\n\n```json\n{json.dumps(lock, indent=2)}\n```\n"


# --- the §7 rule ---------------------------------------------------------------

def test_decide_applies_sign_holm_and_floor():
    r = pd.DataFrame([
        dict(id="A", direction="+", coef=0.10, p_one=0.001),     # confirmed
        dict(id="B", direction="+", coef=0.04, p_one=0.0001),    # below the floor
        dict(id="C", direction="+", coef=-0.10, p_one=0.999),    # wrong sign
        dict(id="D", direction="+", coef=0.08, p_one=0.03),      # p < .05 raw, fails Holm
        dict(id="E", direction="-", coef=-0.07, p_one=0.0005),   # confirmed, negative direction
    ])
    out = analyze.decide(r, alpha=0.05, floor=0.05).set_index("id")
    # Holm by hand, m = 5, sorted p: B .0001x5, E .0005x4, A .001x3, D .03x2, C .999x1
    assert out["p_holm"].to_dict() == pytest.approx(
        dict(A=0.003, B=0.0005, C=0.999, D=0.06, E=0.002))
    assert out["confirmed"].to_dict() == dict(A=True, B=False, C=False, D=False, E=True)
    assert not out.loc["B", "floor_pass"] and out.loc["B", "holm_pass"]
    assert not out.loc["C", "same_sign"]
    assert not out.loc["D", "holm_pass"] and out.loc["D", "floor_pass"]


# --- lock parsing ---------------------------------------------------------------

def test_read_lock():
    lock = analyze.read_lock(lock_text())
    assert lock["hypotheses"][0]["feature"] == "miss_PTS_share"
    analyze.check_lock_matches_code(lock)


def test_read_lock_refuses_a_draft():
    with pytest.raises(analyze.LockError, match="draft"):
        analyze.read_lock(lock_text(status="draft"))
    assert analyze.read_lock(lock_text(status="draft"), require_locked=False)


@pytest.mark.parametrize("text, match", [
    ("no block here", "exactly one"),
    (lock_text() + lock_text(), "exactly one"),
    (lock_text(hypotheses=[dict(id="H1", stat="PTS", feature="arena_team_id", direction="+",
                                explore_coef=0.1, explore_ci=[0, 0.2])]), "not a locked-model"),
    (lock_text(hypotheses=[dict(id="H1", stat="REB", feature="miss_PTS_share", direction="+",
                                explore_coef=0.1, explore_ci=[0, 0.2])]), "not a locked-model"),
    (lock_text(hypotheses=[dict(id="H1", stat="PTS", feature="b2b", direction="up",
                                explore_coef=0.1, explore_ci=[0, 0.2])]), "not a locked-model"),
])
def test_read_lock_rejects(text, match):
    with pytest.raises(analyze.LockError, match=match):
        analyze.read_lock(text)


def test_lock_must_match_the_code():
    lock = analyze.read_lock(lock_text())
    lock["model"]["common_drivers"] = lock["model"]["common_drivers"][:-1]
    with pytest.raises(analyze.LockError, match="common_drivers"):
        analyze.check_lock_matches_code(lock)
    lock = analyze.read_lock(lock_text())
    lock["practical_floor_sd"] = 0.02
    with pytest.raises(analyze.LockError, match="floor"):
        analyze.check_lock_matches_code(lock)


# --- §8 refusals, in a scratch repo ------------------------------------------------

CODE = config.LOCKED_CODE_FILES


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                   cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for name in CODE:
        (tmp_path / name).write_text(f"# {name}\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "code")
    return tmp_path


def check(repo):
    return analyze.check_preconditions(repo, repo / "LOCKED_HYPOTHESES.md",
                                       repo / "reports" / "confirm.md", CODE)


def test_refuses_without_a_lock(repo):
    with pytest.raises(analyze.LockError, match="doesn't exist"):
        check(repo)


def test_refuses_an_uncommitted_lock(repo):
    (repo / "LOCKED_HYPOTHESES.md").write_text(lock_text())
    with pytest.raises(analyze.LockError, match="isn't committed"):
        check(repo)
    git(repo, "add", "LOCKED_HYPOTHESES.md")              # staged is still not committed
    with pytest.raises(analyze.LockError, match="uncommitted changes"):
        check(repo)


def test_accepts_a_committed_lock_then_refuses_edits(repo):
    (repo / "LOCKED_HYPOTHESES.md").write_text(lock_text())
    git(repo, "add", "LOCKED_HYPOTHESES.md")
    git(repo, "commit", "-q", "-m", "lock")
    commit = check(repo)
    assert len(commit) == 40

    (repo / "LOCKED_HYPOTHESES.md").write_text(lock_text() + "\nedited\n")
    with pytest.raises(analyze.LockError, match="uncommitted changes"):
        check(repo)
    git(repo, "checkout", "--", "LOCKED_HYPOTHESES.md")

    (repo / "analyze.py").write_text("# changed after the lock\n")       # uncommitted code edit
    with pytest.raises(analyze.LockError, match="code changed since the lock"):
        check(repo)
    git(repo, "commit", "-q", "-am", "post-lock change")                  # committed code edit
    with pytest.raises(analyze.LockError, match="code changed since the lock"):
        check(repo)


def test_refuses_when_the_report_exists(repo):
    (repo / "LOCKED_HYPOTHESES.md").write_text(lock_text())
    git(repo, "add", "LOCKED_HYPOTHESES.md")
    git(repo, "commit", "-q", "-m", "lock")
    (repo / "reports").mkdir()
    (repo / "reports" / "confirm.md").write_text("already looked\n")
    with pytest.raises(analyze.LockError, match="runs once"):
        check(repo)


def test_confirm_refuses_before_reading_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCK_PATH", tmp_path / "LOCKED_HYPOTHESES.md")
    monkeypatch.setattr(config, "CONFIRM_REPORT_PATH", tmp_path / "confirm.md")

    def sealed(*args, **kwargs):
        raise AssertionError("the confirm season was read before the lock checks passed")
    monkeypatch.setattr(features, "build_season", sealed)
    monkeypatch.setattr(pull, "load_raw", sealed)
    with pytest.raises(analyze.LockError, match="doesn't exist"):
        analyze.confirm()


# --- the evaluation path, on the explore season --------------------------------------

@pytest.fixture(scope="module")
def explore_units():
    season = config.SEASON_EXPLORE
    if not all(pull.raw_path(season, k).exists() for k in config.LOG_KINDS):
        pytest.skip(f"{season} not pulled yet")
    return features.build_season(season)


def test_evaluate_locked_on_the_explore_season(explore_units):
    hyps = [dict(id="H1", stat="PTS", feature="miss_PTS_share", direction="+"),
            dict(id="H2", stat="PTS", feature="miss_PTS_share", direction="-"),
            dict(id="H3", stat="PTS", feature="days_rest", direction="+"),
            dict(id="H4", stat="AST", feature="opp_ast_allowed", direction="+")]
    for h in hyps:
        h.update(explore_coef=0.1, explore_ci=[0.0, 0.2])
    lock = analyze.read_lock(lock_text(hypotheses=hyps))
    r = analyze.evaluate_locked(explore_units, lock).set_index("id")
    assert r["confirmed"].to_dict() == dict(H1=True, H2=False, H3=False, H4=True)
    assert not r.loc["H2", "same_sign"]
    # The coefficients are Phase A's: the same design and fit, on the same rows.
    table, _ = analyze.joint_models(explore_units)
    phase_a = table.set_index(["stat", "driver"])["coef"]
    assert r.loc["H1", "coef"] == pytest.approx(phase_a[("PTS", "miss_PTS_share")])
    assert r.loc["H4", "coef"] == pytest.approx(phase_a[("AST", "opp_ast_allowed")])
    text = analyze.write_confirm_report(r.reset_index(), lock, "abc1234", analyze.provenance())
    assert "2 of 4 hypotheses confirmed" in text and "`abc1234`" in text
    assert np.isfinite(r["p_one"]).all()

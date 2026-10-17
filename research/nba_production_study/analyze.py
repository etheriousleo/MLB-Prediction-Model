"""
Phase A of NBA_STUDY_SPEC.md (§6): explore 2024-25.

    python analyze.py --phase explore     # reports/explore_2024-25.md + reports/figures/
    python analyze.py --phase confirm     # Phase B (§8): the one look at 2025-26 (done)
    python analyze.py --phase forward     # v2 (§12): the one look at 2026-27, after it ends

The test suite runs first, and no report is written unless every test passes
(§9). Phase A builds features for 2024-25 only. Phase B reads 2025-26 only
after every lock check passes (see check_preconditions), runs only the
locked models, and writes reports/confirm_2025-26.md exactly once.

What runs (§6), on each stat's eligible rows (elig_S):
1. Base rates: hit rate, mean and median z, and the z distribution.
2. One factor at a time: binary flags as-is, continuous features in
   quintiles. Each bucket gets n, hit rate, mean z, and 95% CIs clustered by
   player.
3. The joint model: OLS of z_S on the drivers plus season_phase, with SEs
   clustered by player, and a logistic companion on hit_S. Its average
   marginal effects give each driver's effect in hit-rate points.
4. Channels: the same design on min_resid (minutes) and on standardized
   rate_resid_S (per-minute rate).
5. Arena: deviation-coded arena fixed effects, a joint Wald F-test, and the
   Oct-Dec vs. Jan-Apr stability correlation.
6. Sensitivity: the joint model without OT games, then without early exits.
7. Benjamini-Hochberg at 10% FDR across every driver coefficient in the
   three z_S joint models.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.stats import norm  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

import config  # noqa: E402
import features  # noqa: E402
import pull  # noqa: E402

STATS = config.STATS
REPORT_PATH = config.REPORTS_DIR / f"explore_{config.SEASON_EXPLORE}.md"

# Chart tokens: the dataviz reference palette, light surface (static PNGs).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"


# ---------------------------------------------------------------------------
# Specification helpers
# ---------------------------------------------------------------------------

def drivers_for(s: str) -> list:
    return [*config.COMMON_DRIVERS, *(d.format(s=s) for d in config.PER_STAT_DRIVERS)]


def template(driver: str, s: str) -> str:
    """'miss_PTS_share' -> 'miss_{s}_share', for looking up targets and labels."""
    return driver.replace(f"_{s}_", "_{s}_").replace(f"_{s}", "_{s}") if s in driver else driver


def label(driver: str, s: str) -> str:
    """Stat-generic display name: miss_PTS_share -> miss_S_share."""
    return template(driver, s).replace("{s}", "S")


def design(df: pd.DataFrame, drivers, *, arenas=None) -> pd.DataFrame:
    """The joint-model design matrix on df's rows: continuous drivers
    standardized within df, binary drivers 0/1, season_phase dummies (the
    earliest phase present is the reference), optional deviation-coded arena
    effects (the last arena is minus the sum of the others), and a constant."""
    cols = {}
    for d in drivers:
        x = df[d].astype(float)
        cols[d] = x if d in config.BINARY_DRIVERS else (x - x.mean()) / x.std()
    X = pd.DataFrame(cols, index=df.index)
    present = [p for p in config.SEASON_PHASE_ORDER if (df["season_phase"] == p).any()]
    for p in present[1:]:
        X[f"phase_{p}"] = (df["season_phase"] == p).astype(float)
    if arenas is not None:
        a = df["arena_team_id"].astype(int)
        for aid in arenas[:-1]:
            X[f"arena_{aid}"] = (a == aid).astype(float) - (a == arenas[-1]).astype(float)
    return sm.add_constant(X, has_constant="add")


def ols(y, X, groups):
    return sm.OLS(y.astype(float), X).fit(cov_type="cluster",
                                          cov_kwds={"groups": np.asarray(groups)})


def standardized(y: pd.Series) -> pd.Series:
    return (y - y.mean()) / y.std()


# ---------------------------------------------------------------------------
# §6.1 base rates and §6.2 one factor at a time
# ---------------------------------------------------------------------------

def base_rates(units) -> pd.DataFrame:
    rows = []
    for s in STATS:
        df = units[units[f"elig_{s}"]]
        rows.append(dict(stat=s, n=len(df), players=df["PLAYER_ID"].nunique(),
                         hit=df[f"hit_{s}"].mean(), mean_z=df[f"z_{s}"].mean(),
                         median_z=df[f"z_{s}"].median(), sd_z=df[f"z_{s}"].std()))
    return pd.DataFrame(rows).set_index("stat")


def buckets(df: pd.DataFrame, feature: str) -> pd.Series:
    """Binary flags as-is, season_phase by month bucket, the discrete counts
    in config.DISCRETE_BUCKETS by value, everything else in quintiles (fewer
    when ties collapse edges). Quintiles are labeled by the values inside them."""
    x = df[feature]
    if feature == "season_phase":
        order = [p for p in config.SEASON_PHASE_ORDER if (x == p).any()]
        return pd.Series(pd.Categorical(x, categories=order, ordered=True), index=df.index)
    if feature in config.BINARY_DRIVERS:
        return x.map({0.0: "0 (no)", 1.0: "1 (yes)"})
    if feature in config.DISCRETE_BUCKETS:
        values = config.DISCRETE_BUCKETS[feature]
        top = values[-1]
        names = [fmt_num(v) for v in values[:-1]] + [f"{fmt_num(top)}+"]
        return pd.Series(pd.Categorical(x.clip(upper=top).map(dict(zip(values, names))),
                                        categories=names, ordered=True), index=df.index)
    codes = pd.qcut(x, config.N_QUANTILE_BUCKETS, labels=False, duplicates="drop")
    names = {}
    for c in sorted(codes.dropna().unique()):
        inside = x[codes == c]
        lo, hi = inside.min(), inside.max()
        names[c] = f"Q{int(c) + 1}: {fmt_num(lo)}" + ("" if lo == hi else f" to {fmt_num(hi)}")
    return codes.map(names)


def fmt_num(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.3g}"


def clustered_mean(y: pd.Series, groups: pd.Series):
    """Mean and its 95% CI, with the SE clustered by player."""
    res = sm.OLS(y.astype(float).to_numpy(), np.ones(len(y))).fit(
        cov_type="cluster", cov_kwds={"groups": groups.to_numpy()})
    lo, hi = res.conf_int(alpha=1 - config.CI_LEVEL)[0]
    return res.params[0], lo, hi


def one_factor(units, s: str, feature: str) -> pd.DataFrame:
    df = units[units[f"elig_{s}"]]
    base_hit = df[f"hit_{s}"].mean()
    rows = []
    for name, grp in df.groupby(buckets(df, feature), observed=True, sort=True):
        mz, mz_lo, mz_hi = clustered_mean(grp[f"z_{s}"], grp["PLAYER_ID"])
        hr, hr_lo, hr_hi = clustered_mean(grp[f"hit_{s}"], grp["PLAYER_ID"])
        rows.append(dict(bucket=name, n=len(grp), hit=hr, hit_lo=hr_lo, hit_hi=hr_hi,
                         vs_base=hr - base_hit, mean_z=mz, z_lo=mz_lo, z_hi=mz_hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §6.3-6.4 joint models and channels, §6.7 BH
# ---------------------------------------------------------------------------

def channel(min_coef, min_p, rate_coef, rate_p) -> str:
    m, r = min_p < config.CHANNEL_ALPHA, rate_p < config.CHANNEL_ALPHA
    if m and r:
        return "both" if np.sign(min_coef) == np.sign(rate_coef) else "offsetting"
    return "minutes" if m else "rate" if r else "neither"


def joint_models(units):
    """Fit every §6.3-6.4 model; return (tidy driver table, fitted-model details)."""
    rows, detail = [], {}
    swap_in, swap_out = config.USAGE_SWAP
    for s in STATS:
        df = units[units[f"elig_{s}"]]
        drivers = drivers_for(s)
        X, g = design(df, drivers), df["PLAYER_ID"]
        main = ols(df[f"z_{s}"], X, g)
        logit = sm.Logit(df[f"hit_{s}"].astype(float), X).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": g.to_numpy()})
        me = logit.get_margeff(at="overall", dummy=True).summary_frame()
        minutes = ols(df["min_resid"], X, g)
        rdf = df[df[f"rate_resid_{s}"].notna()]
        rate = ols(standardized(rdf[f"rate_resid_{s}"]), design(rdf, drivers), rdf["PLAYER_ID"])
        swapped = [swap_in if d == swap_out.format(s=s) else d for d in drivers]
        usage = ols(df[f"z_{s}"], design(df, swapped), g)

        ci = main.conf_int(alpha=1 - config.CI_LEVEL)
        exog = list(X.columns)
        vif = {d: variance_inflation_factor(X.to_numpy(), exog.index(d)) for d in drivers}
        for d in drivers:
            rows.append(dict(
                stat=s, driver=d, label=label(d, s),
                target=config.HYPOTHESIS_TARGETS.get(template(d, s), ""),
                binary=d in config.BINARY_DRIVERS,
                sd=np.nan if d in config.BINARY_DRIVERS else df[d].std(),
                coef=main.params[d], lo=ci.loc[d, 0], hi=ci.loc[d, 1], p=main.pvalues[d],
                ame=100 * me.loc[d, "dy/dx"], ame_lo=100 * me.loc[d, "Conf. Int. Low"],
                ame_hi=100 * me.loc[d, "Cont. Int. Hi."], ame_p=me.loc[d, "Pr(>|z|)"],
                min_coef=minutes.params[d], min_p=minutes.pvalues[d],
                rate_coef=rate.params[d], rate_p=rate.pvalues[d],
                vif=vif[d]))
        detail[s] = dict(df=df, drivers=drivers, main=main, logit=logit, minutes=minutes,
                         rate=rate, rate_n=len(rdf), usage=usage, swap_out=swap_out.format(s=s))

    table = pd.DataFrame(rows)
    table["channel"] = [channel(*v) for v in table[["min_coef", "min_p", "rate_coef", "rate_p"]].to_numpy()]
    reject, q, _, _ = multipletests(table["p"], alpha=config.BH_FDR, method="fdr_bh")
    table["bh"], table["q"] = reject, q
    table["floor"] = table["coef"].abs() >= config.PRACTICAL_FLOOR_SD
    return table, detail


# ---------------------------------------------------------------------------
# §6.5 arena test, §6.6 sensitivity
# ---------------------------------------------------------------------------

def arena_effects(df, s, drivers, arenas):
    """Fit stat s's joint model with deviation-coded arena effects; return
    the fit, its design, and all 30 effects (the last is minus the sum of
    the other 29)."""
    X = design(df, drivers, arenas=arenas)
    res = ols(df[f"z_{s}"], X, df["PLAYER_ID"])
    eff = {aid: res.params[f"arena_{aid}"] for aid in arenas[:-1]}
    eff[arenas[-1]] = -sum(eff.values())
    return res, X, pd.Series(eff)


def arena_test(units):
    out = {}
    for s in STATS:
        df = units[units[f"elig_{s}"]]
        drivers = drivers_for(s)
        arenas = sorted(int(a) for a in df["arena_team_id"].unique())
        res, X, full = arena_effects(df, s, drivers, arenas)
        cols = [c for c in X.columns if c.startswith("arena_")]
        R = np.zeros((len(cols), X.shape[1]))
        for i, c in enumerate(cols):
            R[i, X.columns.get_loc(c)] = 1.0
        ft = res.f_test(R)
        month = df["GAME_DATE"].dt.month
        halves = {}
        for name, months in (("early", config.ARENA_SPLIT_EARLY_MONTHS),
                             ("late", config.ARENA_SPLIT_LATE_MONTHS)):
            halves[name] = arena_effects(df[month.isin(months)], s, drivers, arenas)[2]
        out[s] = dict(F=float(np.squeeze(ft.fvalue)), df_num=int(ft.df_num),
                      df_denom=float(ft.df_denom), p=float(np.squeeze(ft.pvalue)), effects=full,
                      early=halves["early"], late=halves["late"],
                      r=halves["early"].corr(halves["late"]),
                      rho=halves["early"].corr(halves["late"], method="spearman"),
                      n_early=int(month.isin(config.ARENA_SPLIT_EARLY_MONTHS).sum()),
                      n_late=int(month.isin(config.ARENA_SPLIT_LATE_MONTHS).sum()))
    return out


def sensitivity(units) -> pd.DataFrame:
    rows = []
    for s in STATS:
        base = units[units[f"elig_{s}"]]
        drivers = drivers_for(s)
        for run, df in (("no OT games", base[base["ot_game"] == 0]),
                        ("no early exits", base[base["early_exit"] == 0])):
            res = ols(df[f"z_{s}"], design(df, drivers), df["PLAYER_ID"])
            for d in drivers:
                rows.append(dict(stat=s, driver=d, run=run, n=len(df),
                                 coef=res.params[d], p=res.pvalues[d]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def fig_z_distributions(units, rates, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, facecolor=SURFACE)
    bins = np.arange(-4, 6.01, 0.25)
    for ax, s in zip(axes, STATS):
        z = units.loc[units[f"elig_{s}"], f"z_{s}"].clip(-4, 6)
        ax.hist(z, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=1)
        ax.axvline(rates.loc[s, "mean_z"], color=INK_2, linewidth=1.2)
        ax.axvline(rates.loc[s, "median_z"], color=INK_2, linewidth=1.2, linestyle="--")
        top = ax.get_ylim()[1]
        backing = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
        ax.text(max(rates.loc[s, ["mean_z", "median_z"]]) + 0.3, top * 0.92,
                f"mean {rates.loc[s, 'mean_z']:+.2f} (solid)", color=INK, fontsize=8, bbox=backing)
        ax.text(min(rates.loc[s, ["mean_z", "median_z"]]) - 0.3, top * 0.92,
                f"median {rates.loc[s, 'median_z']:+.2f} (dashed)", color=INK, fontsize=8,
                ha="right", bbox=backing)
        ax.set_title(f"{s}: hit rate {100 * rates.loc[s, 'hit']:.1f}%", fontsize=10, loc="left")
        ax.set_xlabel("z = (stat - own baseline) / own SD")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_ylabel("player-games")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_forest(table, path):
    labels = [label(d, "PTS") for d in drivers_for("PTS")]       # same order for every stat
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 6.2), sharey=True, facecolor=SURFACE)
    for ax, s in zip(axes, STATS):
        t = table[table["stat"] == s].set_index("label").loc[labels]
        for yi, (_, r) in zip(y, t.iterrows()):
            color = SERIES if r["bh"] else MUTED
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(r["coef"], yi, "o", markersize=6, markeredgewidth=1.5, markeredgecolor=color,
                    markerfacecolor=color if r["bh"] else SURFACE)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        for f in (-config.PRACTICAL_FLOOR_SD, config.PRACTICAL_FLOOR_SD):
            ax.axvline(f, color=BASELINE, linewidth=0.8, linestyle=":")
        ax.set_title(s, fontsize=10, loc="left")
        ax.set_xlabel("effect on z (SD units)")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    handles = [plt.Line2D([], [], marker="o", color=SERIES, markerfacecolor=SERIES, linewidth=2,
                          label="passes BH at 10% FDR"),
               plt.Line2D([], [], marker="o", color=MUTED, markerfacecolor=SURFACE, linewidth=2,
                          label="does not pass"),
               plt.Line2D([], [], color=BASELINE, linestyle=":", label=f"±{config.PRACTICAL_FLOOR_SD} SD floor")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_arena(arena, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, s in zip(axes, STATS):
        a = arena[s]
        lim = 1.1 * max(a["early"].abs().max(), a["late"].abs().max())
        ax.plot([-lim, lim], [-lim, lim], color=BASELINE, linewidth=0.8, linestyle="--")
        ax.axhline(0, color=GRID, linewidth=0.6)
        ax.axvline(0, color=GRID, linewidth=0.6)
        ax.plot(a["early"], a["late"], "o", markersize=6, color=SERIES,
                markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{s}: r = {a['r']:+.2f}", fontsize=10, loc="left")
        ax.set_xlabel("arena effect, Oct-Dec (z)")
        _style(ax)
    axes[0].set_ylabel("arena effect, Jan-Apr (z)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def run_tests() -> None:
    """§9: all tests must pass before any report is written."""
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.STUDY_DIR,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-2000:], sep="\n")
        raise SystemExit("Tests failed; no report written (spec §9).")
    print("  tests: " + proc.stdout.strip().splitlines()[-1])


def provenance(season: str = config.SEASON_EXPLORE) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=config.STUDY_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))["files"]
    return dict(commit=git("rev-parse", "--short", "HEAD"),
                # Code and data only: the reports folder holds this run's own outputs.
                dirty=bool(git("status", "--porcelain", "--", ".", ":(exclude)reports")),
                raw={k: v["sha256"][:12] for k, v in manifest.items() if k.startswith(season)},
                when=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                versions=f"Python {sys.version.split()[0]}, pandas {pd.__version__}, "
                         f"numpy {np.__version__}, statsmodels {sm.__version__}")


def driver_table_md(t: pd.DataFrame) -> list:
    """The headline table for one stat, ranked by absolute effect."""
    t = t.reindex(t["coef"].abs().sort_values(ascending=False).index)
    out = ["| # | Driver | Target | Effect on z [95% CI] | Hit-rate pts [95% CI] | "
           "Minutes / SD | Rate (SD) / SD | Channel | q (BH) | BH 10% | ≥ floor |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(t.iterrows(), start=1):
        unit = "flag" if r["binary"] else f"1 SD = {fmt_num(r['sd'])}"
        out.append(
            f"| {i} | `{r['driver']}` ({unit}) | {r['target']} | "
            f"{r['coef']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
            f"{r['ame']:+.2f} [{r['ame_lo']:+.2f}, {r['ame_hi']:+.2f}] | "
            f"{r['min_coef']:+.2f}{stars(r['min_p'])} | {r['rate_coef']:+.3f}{stars(r['rate_p'])} | "
            f"{r['channel']} | {r['q']:.3f} | {'**yes**' if r['bh'] else 'no'} | "
            f"{'yes' if r['floor'] else 'no'} |")
    return out


def write_report(units, rates, table, detail, arena, sens, prov) -> str:
    s_list = ", ".join(STATS)
    n_bh = int(table["bh"].sum())
    cand = table[table["bh"] & table["floor"]]
    md = [
        f"# Phase A: explore {config.SEASON_EXPLORE}",
        "",
        f"`{config.STUDY_VERSION}`, generated {prov['when']} by `analyze.py --phase explore` at commit "
        f"`{prov['commit']}`{' (with uncommitted changes in the study folder)' if prov['dirty'] else ''}. "
        f"Raw data: " + ", ".join(f"`{k}` sha256 `{v}…`" for k, v in prov["raw"].items()) + ". "
        "All tests passed before this report was written.",
        "",
        "Exploration only. 2025-26 has not been touched; it is looked at once, after the lock (§7-8).",
        "",
        "## Summary",
        "",
        f"- **{n_bh} of {len(table)}** driver coefficients pass Benjamini-Hochberg at "
        f"{100 * config.BH_FDR:.0f}% FDR, and **{len(cand)}** of those also clear the proposed "
        f"practical floor of |effect| ≥ {config.PRACTICAL_FLOOR_SD} SD (lock candidates, table below).",
    ]
    for s in STATS:
        a = arena[s]
        md.append(f"- Arena, {s}: F({a['df_num']}, {a['df_denom']:.0f}) = {a['F']:.2f}, "
                  f"p = {a['p']:.2g}; Oct-Dec vs. Jan-Apr effects correlate r = {a['r']:+.2f}.")
    md += [
        "",
        "### Lock candidates (pass BH and the floor)",
        "",
        "| Stat | Driver | Direction | Effect on z [95% CI] | Hit-rate pts | Channel | Target |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in cand.sort_values(["stat", "p"]).iterrows():
        md.append(f"| {r['stat']} | `{r['driver']}` | {'+' if r['coef'] > 0 else '−'} | "
                  f"{r['coef']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['ame']:+.2f} | "
                  f"{r['channel']} | {r['target']} |")
    if cand.empty:
        md.append("| — | none | | | | | |")

    md += [
        "",
        "## How to read the driver tables",
        "",
        "- **Effect on z**: the joint-model OLS coefficient. z is the game's stat minus the "
        "player's own stint baseline, in his own SDs. A continuous driver's effect is per 1 SD of "
        "that driver (its SD on the sample is shown beside it); a flag's effect is on vs. off. "
        "95% CIs are clustered by player.",
        "- **Hit-rate pts**: the logistic companion's average marginal effect on P(stat > "
        "baseline), in percentage points, per 1 SD or for the flag.",
        "- **Minutes / SD**: the same design on `min_resid`, in minutes. **Rate (SD) / SD**: the "
        "same design on standardized `rate_resid_S` (games with MIN ≥ 10). * p < 0.05, ** p < 0.01. "
        f"**Channel** names the one(s) with p < {config.CHANNEL_ALPHA}.",
        f"- **q (BH)** is the Benjamini-Hochberg adjusted p across all {len(table)} driver "
        "coefficients in the three z models. **≥ floor** is |effect| ≥ "
        f"{config.PRACTICAL_FLOOR_SD} SD (the proposed practical floor; Juan confirms it at the lock).",
        "- **Target** is the spec's hypothesis target for that feature (§5.3).",
        "",
        "![Joint-model effects](figures/drivers_forest.png)",
        "",
    ]
    for s in STATS:
        d = detail[s]
        md += [f"## {s}: ranked drivers", "",
               f"{len(d['df']):,} player-games, {d['df']['PLAYER_ID'].nunique()} players "
               f"(rate model: {d['rate_n']:,} games with MIN ≥ {config.RATE_RESID_MIN_MINUTES}). "
               f"OLS R² = {d['main'].rsquared:.3f}.", ""]
        md += driver_table_md(table[table["stat"] == s])
        u = d["usage"]
        md += ["", f"Usage swap: with `miss_usage_share` in place of `{d['swap_out']}`, its effect is "
               f"{u.params['miss_usage_share']:+.3f} [{u.conf_int().loc['miss_usage_share', 0]:+.3f}, "
               f"{u.conf_int().loc['miss_usage_share', 1]:+.3f}], p = {u.pvalues['miss_usage_share']:.2g}.",
               ""]

    md += ["## Base rates (§6.1)", "",
           "Counting stats are right-skewed, so the hit rate sits below 50% and the median z below 0. "
           "Judge every factor against these base rates, not against 50%. Mean z sits slightly "
           "above 0 because eligibility admits players once their minutes reach 20 and a stint "
           "average trails a player who is improving; the joint models' intercept and "
           "season_phase absorb that level.", "",
           "| Stat | Player-games | Players | Hit rate | Mean z | Median z | SD of z |",
           "|---|---|---|---|---|---|---|"]
    for s, r in rates.iterrows():
        md.append(f"| {s} | {int(r['n']):,} | {int(r['players'])} | {100 * r['hit']:.1f}% | {r['mean_z']:+.3f} | "
                  f"{r['median_z']:+.3f} | {r['sd_z']:.3f} |")
    md += ["", "![z distributions](figures/z_distributions.png)", ""]

    md += ["## Arena test (§6.5)", "",
           "Arena fixed effects (home team's TEAM_ID, a proxy for the scorekeeper) added to each "
           "stat's joint model, deviation-coded so each effect is relative to the average arena. "
           "Joint Wald F-test with player-clustered SEs. Stability: the 30 effects estimated on "
           "Oct-Dec games vs. on Jan-Apr games. Expected: real for AST, possibly REB, null for PTS. "
           "A strong PTS effect would mean something else is wrong.", "",
           "| Stat | F | df | p | r (Oct-Dec vs. Jan-Apr) | Spearman | Games early / late | "
           "Spread of effects (SD) |", "|---|---|---|---|---|---|---|---|"]
    for s in STATS:
        a = arena[s]
        md.append(f"| {s} | {a['F']:.2f} | {a['df_num']}, {a['df_denom']:.0f} | {a['p']:.2g} | "
                  f"{a['r']:+.2f} | {a['rho']:+.2f} | {a['n_early']:,} / {a['n_late']:,} | "
                  f"{a['effects'].std():.3f} |")
    abbr = arena_abbreviations()
    md += [""]
    for s in STATS:
        eff = arena[s]["effects"].sort_values()
        lo = ", ".join(f"{abbr.get(i, i)} {v:+.3f}" for i, v in eff.head(3).items())
        hi = ", ".join(f"{abbr.get(i, i)} {v:+.3f}" for i, v in eff.tail(3)[::-1].items())
        md.append(f"- {s}: most generous arenas {hi}; stingiest {lo}.")
    md += ["", "![Arena stability](figures/arena_stability.png)", ""]

    md += ["## Sensitivity (§6.6)", "",
           "Joint-model effects on z, rerun without OT games and without early exits "
           f"(MIN < {config.EARLY_EXIT_FRAC} × base_min). A sign flip is marked ⚠.", ""]
    for s in STATS:
        t = table[table["stat"] == s].set_index("driver")
        sv = sens[sens["stat"] == s].pivot(index="driver", columns="run", values="coef")
        n = sens[sens["stat"] == s].groupby("run")["n"].first()
        md += [f"**{s}** (n: main {len(detail[s]['df']):,}, no OT {n['no OT games']:,}, "
               f"no early exits {n['no early exits']:,})", "",
               "| Driver | Main | No OT games | No early exits |", "|---|---|---|---|"]
        for d in detail[s]["drivers"]:
            m = t.loc[d, "coef"]
            cells = []
            for run in ("no OT games", "no early exits"):
                v = sv.loc[d, run]
                cells.append(f"{v:+.3f}" + (" ⚠" if np.sign(v) != np.sign(m) else ""))
            md.append(f"| `{d}`{' (BH)' if t.loc[d, 'bh'] else ''} | {m:+.3f} | {cells[0]} | {cells[1]} |")
        md.append("")

    md += ["## Model diagnostics", "",
           "Variance inflation factors above 5 (shared variance makes those coefficients partial "
           "effects that are hard to separate):", ""]
    high = table[table["vif"] > 5]
    md += [f"- {r['stat']} `{r['driver']}`: VIF {r['vif']:.1f}" for _, r in high.iterrows()] or ["- none"]
    top = []
    for s in STATS:
        worst = table[table["stat"] == s].nlargest(3, "vif")
        top.append(f"{s}: " + ", ".join(f"`{d}` {v:.1f}" for d, v in zip(worst["driver"], worst["vif"])))
    md += ["", "Highest VIFs per stat: " + "; ".join(top) + ".", ""]

    md += ["## Choices made in this run (for review)", "",
           "- **Teammate shares:** each stat's model carries its own `miss_S_share`; `miss_usage_share` "
           "is fit as a swap-in variant (reported under each table), because the two correlate at "
           "r = 0.995 (PTS), 0.94 (AST) and 0.91 (REB). The spec says \"all features\" (§6.3).",
           "- **Channel models:** `min_resid` is fit on each stat's design and sample rather than "
           "once, so a driver's minutes and rate effects decompose the same model. The spec says "
           "once (§6.4); the common drivers' minutes effects are near-identical across the three fits.",
           "- **BH family:** the driver coefficients of the three z models "
           f"({len(table)} tests). Logistic, channel, sensitivity and arena results are descriptive "
           "and not in the family. season_phase is a control and not in it either.",
           "- **Standardization:** continuous drivers are z-scored within each model's sample; flags "
           "stay 0/1. **Inference:** Wald tests with player-clustered SEs, normal reference.",
           "- **Neutral-site games** are never analysis rows (Juan, 2026-10-17); see "
           "`step1_findings.md`.",
           "- **Fresh absences use the actual box score**: a late scratch isn't knowable before "
           "tip-off. Accepted for a drivers study (§5.3), but a live use would need injury reports.",
           ""]

    md += ["## Appendix: one factor at a time (§6.2)", "",
           "Each feature bucketed on its own (binary as-is, continuous in quintiles; ties can "
           "merge quintiles, and `days_rest` is bucketed by value: 0, 1, 2, 3+). Mean z and hit rate with 95% CIs clustered by player; **vs. base** is "
           "the bucket's hit rate minus the stat's overall hit rate, in points.", ""]
    for s in STATS:
        md += [f"### {s}", ""]
        feats = [*drivers_for(s), config.USAGE_SWAP[0], "season_phase"]
        for f in feats:
            ot = one_factor(units, s, f)
            md += [f"**`{f}`**", "", "| Bucket | n | Hit rate [95% CI] | vs. base | Mean z [95% CI] |",
                   "|---|---|---|---|---|"]
            for _, r in ot.iterrows():
                md.append(f"| {r['bucket']} | {r['n']:,} | {100 * r['hit']:.1f}% "
                          f"[{100 * r['hit_lo']:.1f}, {100 * r['hit_hi']:.1f}] | {100 * r['vs_base']:+.1f} | "
                          f"{r['mean_z']:+.3f} [{r['z_lo']:+.3f}, {r['z_hi']:+.3f}] |")
            md.append("")
    return "\n".join(md) + "\n"


def arena_abbreviations() -> dict:
    _, team = pull.load_raw(config.SEASON_EXPLORE)
    return team.drop_duplicates("TEAM_ID").set_index("TEAM_ID")["TEAM_ABBREVIATION"].to_dict()


def explore() -> int:
    print(f"{config.STUDY_VERSION} Phase A: {config.SEASON_EXPLORE}")
    run_tests()
    units = features.build_season(config.SEASON_EXPLORE)
    rates = base_rates(units)
    table, detail = joint_models(units)
    arena = arena_test(units)
    sens = sensitivity(units)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_z_distributions(units, rates, config.FIGURES_DIR / "z_distributions.png")
    fig_forest(table, config.FIGURES_DIR / "drivers_forest.png")
    fig_arena(arena, config.FIGURES_DIR / "arena_stability.png")
    REPORT_PATH.write_text(write_report(units, rates, table, detail, arena, sens, provenance()),
                           encoding="utf-8")
    print(f"  {int(table['bh'].sum())} of {len(table)} driver coefficients pass BH; "
          f"wrote {REPORT_PATH.relative_to(config.STUDY_DIR)}")
    return 0


# ---------------------------------------------------------------------------
# Phase B (§7-8): the one look at the confirm season
# ---------------------------------------------------------------------------

class LockError(SystemExit):
    """A Phase B precondition failed. Nothing about the confirm season was read."""


def read_lock(text: str, *, require_locked: bool = True) -> dict:
    """The lock file's machine-readable block: its single ```json fence."""
    blocks = re.findall(r"```json\n(.*?)\n```", text, flags=re.S)
    if len(blocks) != 1:
        raise LockError(f"the lock file must contain exactly one ```json block (found {len(blocks)})")
    lock = json.loads(blocks[0])
    if require_locked and lock.get("status") != "locked":
        raise LockError(f"the lock's status is {lock.get('status')!r}, not 'locked'")
    ids = [h["id"] for h in lock["hypotheses"]]
    if len(set(ids)) != len(ids):
        raise LockError("hypothesis IDs must be unique")
    for h in lock["hypotheses"]:
        allowed = [*lock["model"]["common_drivers"],
                   *(d.format(s=h["stat"]) for d in lock["model"]["per_stat_drivers"])]
        if h["stat"] not in STATS or h["feature"] not in allowed or h["direction"] not in ("+", "-"):
            raise LockError(f"{h['id']}: not a locked-model coefficient: {h}")
    return lock


def check_lock_matches_code(lock: dict, version: str = config.STUDY_VERSION) -> None:
    """The lock's model, version and thresholds must be the ones this code runs."""
    m = lock["model"]
    expected = dict(common_drivers=list(config.COMMON_DRIVERS),
                    per_stat_drivers=list(config.PER_STAT_DRIVERS),
                    binary_drivers=list(config.BINARY_DRIVERS))
    for key, value in expected.items():
        if m[key] != value:
            raise LockError(f"lock model.{key} differs from config: {m[key]} vs {value}")
    if lock["study_version"] != version:
        raise LockError(f"lock is for {lock['study_version']}, this phase runs {version}")
    if lock["practical_floor_sd"] != config.PRACTICAL_FLOOR_SD or \
            lock["alpha_one_sided"] != config.CONFIRM_ALPHA:
        raise LockError("the lock's floor or alpha differs from config.py")


def check_preconditions(study_dir, lock_path, report_path, code_files) -> str:
    """Spec §8's refusals, checked before anything reads the confirm season.
    Returns the commit hash of the lock file."""
    if report_path.exists():
        raise LockError(f"{report_path.name} already exists. Phase B runs once (§8).")
    if not lock_path.exists():
        raise LockError(f"{lock_path.name} doesn't exist. Juan approves and commits it first (§7).")

    def git(*args):
        return subprocess.run(["git", *args], cwd=study_dir, capture_output=True, text=True)

    if git("ls-files", "--error-unmatch", lock_path.name).returncode != 0:
        raise LockError(f"{lock_path.name} isn't committed.")
    if git("status", "--porcelain", "--", lock_path.name).stdout.strip():
        raise LockError(f"{lock_path.name} has uncommitted changes.")
    commit = git("log", "-1", "--format=%H", "--", lock_path.name).stdout.strip()
    if not commit:
        raise LockError(f"{lock_path.name} has no commit.")
    changed = git("diff", "--name-only", commit, "--", *code_files).stdout.split()
    if changed:
        raise LockError(f"code changed since the lock commit {commit[:7]}: {changed}. "
                        "Phase B runs only the locked code.")
    return commit


def decide(results: pd.DataFrame, alpha: float, floor: float) -> pd.DataFrame:
    """The §7 confirmation rule, fixed before Phase A. A hypothesis is
    confirmed only if (1) its 2025-26 coefficient has the locked sign, (2) its
    one-sided p in the locked direction is below alpha after Holm across the
    locked list, and (3) the point estimate is at least the floor in the
    locked direction."""
    r = results.copy()
    sign = np.where(r["direction"] == "+", 1.0, -1.0)
    r["p_holm"] = multipletests(r["p_one"], method="holm")[1]
    r["same_sign"] = np.sign(r["coef"]) == sign
    r["holm_pass"] = r["p_holm"] < alpha
    r["floor_pass"] = sign * r["coef"] >= floor
    r["confirmed"] = r["same_sign"] & r["holm_pass"] & r["floor_pass"]
    return r


def evaluate_locked(units: pd.DataFrame, lock: dict) -> pd.DataFrame:
    """Fit each stat's locked joint model once and read off the locked
    coefficients. Same design() and ols() as Phase A; nothing else is fit."""
    m, rows = lock["model"], []
    for s in STATS:
        hyps = [h for h in lock["hypotheses"] if h["stat"] == s]
        if not hyps:
            continue
        df = units[units[f"elig_{s}"]]
        drivers = [*m["common_drivers"], *(d.format(s=s) for d in m["per_stat_drivers"])]
        res = ols(df[f"z_{s}"], design(df, drivers), df["PLAYER_ID"])
        ci = res.conf_int(alpha=1 - config.CI_LEVEL)
        for h in hyps:
            f = h["feature"]
            sign = 1.0 if h["direction"] == "+" else -1.0
            rows.append(dict(id=h["id"], stat=s, feature=f, direction=h["direction"],
                             n=len(df), players=df["PLAYER_ID"].nunique(),
                             coef=res.params[f], lo=ci.loc[f, 0], hi=ci.loc[f, 1],
                             p_one=float(norm.sf(sign * res.params[f] / res.bse[f])),
                             explore_coef=h["explore_coef"], explore_lo=h["explore_ci"][0],
                             explore_hi=h["explore_ci"][1],
                             confirm_coef=h.get("confirm_coef", np.nan),
                             confirm_lo=h.get("confirm_ci", [np.nan, np.nan])[0],
                             confirm_hi=h.get("confirm_ci", [np.nan, np.nan])[1]))
    return decide(pd.DataFrame(rows), lock["alpha_one_sided"], lock["practical_floor_sd"])


# The two locked looks. Each runs once, against its own lock file.
LOCKED_PHASES = {
    "confirm": dict(season=config.SEASON_CONFIRM, version=config.STUDY_VERSION,
                    lock="LOCK_PATH", report="CONFIRM_REPORT_PATH", title="Phase B: confirm",
                    look="the study's one look", cite="§8"),
    "forward": dict(season=config.SEASON_FORWARD, version=config.FORWARD_STUDY_VERSION,
                    lock="FORWARD_LOCK_PATH", report="FORWARD_REPORT_PATH",
                    title="v2 forward check:", look="v2's one look", cite="§12"),
}


def write_locked_report(r: pd.DataFrame, lock: dict, lock_commit: str, prov: dict,
                        phase: str = "confirm") -> str:
    ph = LOCKED_PHASES[phase]
    season, n_ok = ph["season"], int(r["confirmed"].sum())
    has_confirm = r["confirm_coef"].notna().any() if "confirm_coef" in r else False
    md = [
        f"# {ph['title']} {season}",
        "",
        f"`{ph['version']}`, generated {prov['when']} by `analyze.py --phase {phase}` at commit "
        f"`{prov['commit']}`. Lock: `{getattr(config, ph['lock']).name}` at commit `{lock_commit}`, "
        f"approved by {lock.get('approved_by', '?')} on {lock.get('approved_on', '?')}. Raw data: "
        + ", ".join(f"`{k}` sha256 `{v}…`" for k, v in prov["raw"].items()) + ". "
        f"{prov.get('versions', '')}. All tests passed before this report was written.",
        "",
        f"This is {ph['look']} at {season} ({ph['cite']}). Only the locked models ran: "
        "no new features, buckets, or exploratory tables.",
        "",
        f"## Result: {n_ok} of {len(r)} hypotheses confirmed",
        "",
        "A hypothesis is confirmed only if all three hold (§7, fixed before Phase A): the "
        f"{season} effect has the locked sign; its one-sided p in the locked direction "
        f"is below {lock['alpha_one_sided']} after Holm correction across all {len(r)} locked "
        f"hypotheses; and the point estimate is at least {lock['practical_floor_sd']} SD in the "
        "locked direction.",
        "",
    ]
    prior = [(config.SEASON_EXPLORE, "explore")] + ([(config.SEASON_CONFIRM, "confirm")] if has_confirm else [])
    head = " | ".join(f"{sn} effect on z [95% CI]" for sn, _ in prior)
    md += [f"| ID | Stat | Feature | Direction | {head} | {season} effect on z [95% CI] | "
           "One-sided p | Holm p | ≥ floor | Verdict |",
           "|---|---|---|---|" + "---|" * len(prior) + "---|---|---|---|---|"]
    for _, h in r.iterrows():
        cells = " | ".join(f"{h[f'{k}_coef']:+.3f} [{h[f'{k}_lo']:+.3f}, {h[f'{k}_hi']:+.3f}]"
                           for _, k in prior)
        md.append(
            f"| {h['id']} | {h['stat']} | `{h['feature']}` | {h['direction']} | {cells} | "
            f"{h['coef']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}] | {h['p_one']:.2g} | "
            f"{h['p_holm']:.2g} | {'yes' if h['floor_pass'] else 'no'} | "
            f"{'**confirmed**' if h['confirmed'] else 'not confirmed'} |")
    md += ["", "## Samples", "", "| Stat | Player-games | Players |", "|---|---|---|"]
    for s, g in r.groupby("stat", sort=False):
        md.append(f"| {s} | {int(g['n'].iloc[0]):,} | {int(g['players'].iloc[0])} |")
    md += ["", "## Locked model", "",
           "Each stat's joint model, exactly as locked: OLS of `z_S` on "
           + ", ".join(f"`{d}`" for d in lock["model"]["common_drivers"]) + ", "
           + ", ".join(f"`{d.replace('{s}', 'S')}`" for d in lock["model"]["per_stat_drivers"])
           + ", season_phase dummies and a constant, on the season's `elig_S` rows. Continuous "
           "drivers are standardized within that sample; flags stay 0/1. SEs are clustered by "
           "player. Only the locked coefficients are reported.", ""]
    return "\n".join(md) + "\n"


def write_confirm_report(r, lock, lock_commit, prov) -> str:
    return write_locked_report(r, lock, lock_commit, prov, "confirm")


def run_locked_phase(phase: str) -> int:
    """One locked look: every refusal first, then the tests, then the season's
    data, the locked models, and the report, which can be written only once."""
    ph = LOCKED_PHASES[phase]
    lock_path, report_path = getattr(config, ph["lock"]), getattr(config, ph["report"])
    print(f"{ph['version']} {ph['title']} {ph['season']}")
    lock_commit = check_preconditions(config.STUDY_DIR, lock_path, report_path,
                                      config.LOCKED_CODE_FILES)
    lock = read_lock(lock_path.read_text(encoding="utf-8"))
    check_lock_matches_code(lock, ph["version"])
    print(f"  lock: {lock_path.name} at {lock_commit[:7]}, {len(lock['hypotheses'])} hypotheses")
    run_tests()
    units = features.build_season(ph["season"], after_lock=True)
    results = evaluate_locked(units, lock)
    report_path.write_text(write_locked_report(results, lock, lock_commit,
                                               provenance(ph["season"]), phase), encoding="utf-8")
    print(f"  {int(results['confirmed'].sum())} of {len(results)} confirmed; "
          f"wrote {report_path.relative_to(config.STUDY_DIR)}")
    return 0


def confirm() -> int:
    return run_locked_phase("confirm")


def forward() -> int:
    return run_locked_phase("forward")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NBA production study analysis (spec §6, §8).")
    parser.add_argument("--phase", choices=("explore", "confirm", "forward"), required=True)
    args = parser.parse_args(argv)
    return {"explore": explore, "confirm": confirm, "forward": forward}[args.phase]()


if __name__ == "__main__":
    sys.exit(main())

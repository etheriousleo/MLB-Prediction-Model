"""
All constants for the NBA player production study (`nba-study-v1`).

NBA_STUDY_SPEC.md §2: every season, threshold, floor and the shrinkage
constant lives here, and nothing is tunable from the command line. Section
references (§) point at the spec. Change a value only before any analysis
has run; after the Phase A report exists, a change here is a new study
version, not an edit.
"""
from __future__ import annotations

from pathlib import Path

STUDY_VERSION = "nba-study-v1"

# --- §3 Seasons -------------------------------------------------------------
# 2024-25 is explored freely. 2025-26 is touched once, after the lock (§7):
# until then nothing but row counts and pass/fail hygiene is printed for it.
SEASON_EXPLORE = "2024-25"
SEASON_CONFIRM = "2025-26"
SEASONS = (SEASON_EXPLORE, SEASON_CONFIRM)
SEASON_TYPE = "Regular Season"

# --- §4 Data pull -----------------------------------------------------------
API_TIMEOUT = 60                  # seconds per request
API_RETRIES = 3                   # retries after the first attempt (4 tries max)
API_BACKOFF_SECONDS = (5, 10, 20)  # wait before retry 1, 2, 3
API_PAUSE_SECONDS = 2.0           # between consecutive calls
LOG_KINDS = {"player": "P", "team": "T"}  # file label -> player_or_team_abbreviation

# A 30-team, 82-game schedule. The NBA Cup championship game is not a
# regular-season game (it doesn't count in standings or stats), so it must
# not appear: if it did, the two finalists would show 83 games and the season
# 1,231. Regular-season GAME_IDs carry the "002" prefix; the Cup final is "006".
N_TEAMS = 30
GAMES_PER_TEAM = 82
EXPECTED_GAMES = N_TEAMS * GAMES_PER_TEAM // 2   # 1,230
REGULAR_SEASON_GAME_ID_PREFIX = "002"

# Neutral-site games. stats.nba.com marks both team rows with "@" (no "vs."),
# so neither team is at home, and the home team's scorekeeper isn't there.
# The step-1 sanity check found exactly these (2026-10-17); the dates and teams
# match the published international games and NBA Cup semifinals. The check
# fails if any other game looks like this, or if a listed one doesn't.
NEUTRAL_SITE_GAMES = {
    "2024-25": {
        "0022400147": "Mexico City, 2024-11-02, MIA-WAS",
        "0022400621": "Paris, 2025-01-23, IND-SAS",
        "0022400633": "Paris, 2025-01-25, IND-SAS",
        "0022401229": "Las Vegas, NBA Cup semifinal, 2024-12-14, ATL-MIL",
        "0022401230": "Las Vegas, NBA Cup semifinal, 2024-12-14, HOU-OKC",
    },
    "2025-26": {
        "0022500147": "Mexico City, 2025-11-01, DAL-DET",
        "0022500578": "Berlin, 2026-01-15, MEM-ORL",
        "0022500602": "London, 2026-01-18, MEM-ORL",
        "0022501229": "Las Vegas, NBA Cup semifinal, 2025-12-13, NYK-ORL",
        "0022501230": "Las Vegas, NBA Cup semifinal, 2025-12-13, OKC-SAS",
    },
}

# Columns the study reads from each log; the pull fails loudly if any is missing.
REQUIRED_COLUMNS = {
    "player": ("PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN",
               "FGA", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "PTS"),
    "team": ("TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN", "FGM", "FGA",
             "FG3M", "FG3A", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "PTS"),
}

# --- §5.1 Unit and eligibility ----------------------------------------------
STATS = ("PTS", "REB", "AST")
MIN_PRIOR_GAMES = 10              # prior games in the stint
MIN_BASE_MINUTES = 20.0           # pre-game average minutes
STAT_FLOORS = {"PTS": 8.0, "REB": 3.0, "AST": 2.0}   # on base_S, per stat

# --- §5.2 Baseline and outcomes ---------------------------------------------
SD_FLOOR = 1.0                    # floor on sd_S
RATE_RESID_MIN_MINUTES = 10       # rate_resid_S only when MIN >= this

# --- §5.3 Pre-game features --------------------------------------------------
SHRINK_K = 5                      # x = (n*team + k*league) / (n + k)
POSS_FTA_COEF = 0.44              # POSS = FGA + 0.44*FTA - OREB + TOV
REGULATION_TEAM_MIN = 240         # 5 players x 48 minutes
FORM_WINDOW = 5                   # last5_dev_S, last5_min_dev
FRESH_ABSENCE_LOOKBACK = 5        # regular must have played in one of the team's last 5
REGULAR_MIN_PRIOR_GAMES = 10      # "regular" teammate: same bar as eligibility
REGULAR_MIN_BASE_MINUTES = 20.0

# season_phase is a control, not a hypothesis. October is folded into
# November: no October game can be eligible (10 prior games in the stint).
SEASON_PHASE_BY_MONTH = {10: "Oct-Nov", 11: "Oct-Nov", 12: "Dec", 1: "Jan",
                         2: "Feb", 3: "Mar", 4: "Apr"}

# --- §5.4 Post-game diagnostics (never driver inputs) -----------------------
EARLY_EXIT_FRAC = 0.5             # early_exit: MIN < 0.5 * base_min

# --- §6 Phase A --------------------------------------------------------------
N_QUANTILE_BUCKETS = 5
# Discrete counts get one bucket per value instead of quintiles, with the
# last value top-coded (3 -> "3+"). 80% of games follow 0 or 1 rest days,
# so quintiles of days_rest collapse to two buckets.
DISCRETE_BUCKETS = {"days_rest": (0, 1, 2, 3)}
CI_LEVEL = 0.95
BH_FDR = 0.10
ARENA_SPLIT_EARLY_MONTHS = (10, 11, 12)   # stability check: Oct-Dec ...
ARENA_SPLIT_LATE_MONTHS = (1, 2, 3, 4)    # ... vs Jan-Apr

# The joint model (§6.3): drivers in every stat's model, plus two per stat
# ("{s}" -> PTS, REB, AST). season_phase enters as a control (dummies,
# Oct-Nov as the reference), never as a driver. Arena enters only in the
# arena test (§6.5). Continuous drivers are standardized within each model's
# sample; the BINARY_DRIVERS stay 0/1.
COMMON_DRIVERS = ("is_home", "days_rest", "b2b", "opp_b2b", "three_in_four",
                  "opp_pace", "opp_drtg", "opp_efg_allowed", "opp_ast_allowed",
                  "opp_fg_pct", "opp_3pa_rate", "opp_oreb_pct", "opp_dreb_pct",
                  "net_gap", "abs_net_gap", "last5_min_dev")
PER_STAT_DRIVERS = ("miss_{s}_share", "last5_dev_{s}")
BINARY_DRIVERS = ("is_home", "b2b", "opp_b2b", "three_in_four")
SEASON_PHASE_ORDER = ("Oct-Nov", "Dec", "Jan", "Feb", "Mar", "Apr")
# Departure from "all features" (§6.3), for Juan's review at the Phase A stop:
# miss_usage_share correlates with the per-stat shares at r = 0.995 (PTS),
# 0.94 (AST) and 0.91 (REB) on eligible 2024-25 rows. With both in one model,
# their separate coefficients mean nothing. So each stat's model carries its
# own miss_{s}_share, and miss_usage_share is fit as a swap-in variant.
USAGE_SWAP = ("miss_usage_share", "miss_{s}_share")   # (swap in, swap out)

# The spec's hypothesis targets (§5.3), shown beside each Phase A result.
HYPOTHESIS_TARGETS = {
    "is_home": "All", "days_rest": "All", "b2b": "Minutes", "opp_b2b": "All",
    "three_in_four": "Minutes", "opp_pace": "All", "opp_drtg": "PTS",
    "opp_efg_allowed": "PTS, AST", "opp_ast_allowed": "AST", "opp_fg_pct": "REB",
    "opp_3pa_rate": "REB", "opp_oreb_pct": "REB", "opp_dreb_pct": "REB",
    "miss_usage_share": "PTS, AST", "miss_{s}_share": "Per stat",
    "net_gap": "Minutes", "abs_net_gap": "Minutes",
    "last5_dev_{s}": "Mean reversion vs. hot hand", "last5_min_dev": "Minutes",
    "arena_team_id": "AST, REB (PTS is the control)",
}
CHANNEL_ALPHA = 0.05              # a channel (minutes / rate) counts when its p < this

# --- §7 Lock / §8 Phase B ----------------------------------------------------
PRACTICAL_FLOOR_SD = 0.05         # PROPOSED; Juan confirms or changes it at the lock
CONFIRM_ALPHA = 0.05              # one-sided, after Holm across the locked list

# --- Paths -------------------------------------------------------------------
STUDY_DIR = Path(__file__).resolve().parent
RAW_DIR = STUDY_DIR / "data" / "raw"
FEATURES_DIR = STUDY_DIR / "data" / "features"
REPORTS_DIR = STUDY_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MANIFEST_PATH = RAW_DIR / "manifest.json"
PULL_REPORT_PATH = REPORTS_DIR / "pull_sanity.md"
LOCK_PATH = STUDY_DIR / "LOCKED_HYPOTHESES.md"

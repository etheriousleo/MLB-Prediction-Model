"""
nba_teams.py — the NBA suite's one team table.

Shared by nba_app.py and scripts/snapshot_closes_nba.py so the ESPN-ID join
is defined in exactly one place. Every team join in the NBA app is by ESPN
TEAM ID (the CFB app learned that display names differ across ESPN
endpoints; IDs don't). The Odds API knows nothing about ESPN IDs, so this
file also carries the name aliases that fold its spellings onto the ID —
the same lesson as the MLB app's canon_team(): the Athletics dropped
"Oakland" and every feed silently missed their games until the name
boundary was folded in one place.

IDs are ESPN's stable NBA team IDs (espn.com/nba/team/_/id/<id>).
"""

NBA_TEAMS = {
    "1":  ("ATL",  "Atlanta Hawks"),
    "2":  ("BOS",  "Boston Celtics"),
    "3":  ("NO",   "New Orleans Pelicans"),
    "4":  ("CHI",  "Chicago Bulls"),
    "5":  ("CLE",  "Cleveland Cavaliers"),
    "6":  ("DAL",  "Dallas Mavericks"),
    "7":  ("DEN",  "Denver Nuggets"),
    "8":  ("DET",  "Detroit Pistons"),
    "9":  ("GS",   "Golden State Warriors"),
    "10": ("HOU",  "Houston Rockets"),
    "11": ("IND",  "Indiana Pacers"),
    "12": ("LAC",  "LA Clippers"),
    "13": ("LAL",  "Los Angeles Lakers"),
    "14": ("MIA",  "Miami Heat"),
    "15": ("MIL",  "Milwaukee Bucks"),
    "16": ("MIN",  "Minnesota Timberwolves"),
    "17": ("BKN",  "Brooklyn Nets"),
    "18": ("NY",   "New York Knicks"),
    "19": ("ORL",  "Orlando Magic"),
    "20": ("PHI",  "Philadelphia 76ers"),
    "21": ("PHX",  "Phoenix Suns"),
    "22": ("POR",  "Portland Trail Blazers"),
    "23": ("SAC",  "Sacramento Kings"),
    "24": ("SA",   "San Antonio Spurs"),
    "25": ("OKC",  "Oklahoma City Thunder"),
    "26": ("UTAH", "Utah Jazz"),
    "27": ("WSH",  "Washington Wizards"),
    "28": ("TOR",  "Toronto Raptors"),
    "29": ("MEM",  "Memphis Grizzlies"),
    "30": ("CHA",  "Charlotte Hornets"),
}

# Spellings seen outside ESPN (The Odds API, books, hand entry). Keys are
# normalised with _norm(). Add here — never special-case at a call site.
_ALIASES = {
    "los angeles clippers": "12", "la clippers": "12", "l.a. clippers": "12",
    "l.a. lakers": "13", "la lakers": "13",
    "ny knicks": "18", "new york knicks": "18",
    "gs warriors": "9", "golden state": "9",
    "okc thunder": "25", "oklahoma city": "25",
    "san antonio": "24", "sa spurs": "24",
    "new orleans": "3", "no pelicans": "3",
    "philadelphia sixers": "20", "philadelphia seventy sixers": "20",
    "portland trailblazers": "22", "portland": "22",
    "washington": "27", "utah": "26", "phoenix": "21",
}


def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    for ch in ".,'’-":
        s = s.replace(ch, "" if ch != "-" else " ")
    return " ".join(s.split())


_BY_NAME = {_norm(full): tid for tid, (_, full) in NBA_TEAMS.items()}
_BY_ABBR = {abb.lower(): tid for tid, (abb, _) in NBA_TEAMS.items()}
_BY_NICK = {_norm(full.split()[-1]): tid for tid, (_, full) in NBA_TEAMS.items()}
_BY_NICK["trail blazers"] = "22"


def team_id_for(name: str) -> str | None:
    """Fold any external spelling of a team onto its ESPN team ID.
    Returns None for anything unrecognised (never guesses)."""
    if name is None:
        return None
    raw = str(name).strip()
    if raw in NBA_TEAMS:
        return raw
    n = _norm(raw)
    if not n:
        return None
    if n in _BY_NAME:
        return _BY_NAME[n]
    if n in _ALIASES:
        return _ALIASES[n]
    if n in _BY_ABBR:
        return _BY_ABBR[n]
    # Nickname fallback: "Boston Celtics", "Celtics", "the celtics" all
    # end with the nickname; nicknames are unique across the league.
    for nick, tid in _BY_NICK.items():
        if n == nick or n.endswith(" " + nick):
            return tid
    return None


def team_name(tid: str) -> str:
    return NBA_TEAMS.get(str(tid), ("", str(tid)))[1]


def team_abbr(tid: str) -> str:
    return NBA_TEAMS.get(str(tid), (str(tid), ""))[0]

import re


def get_season(gameVersion: str) -> int:
    regex = re.compile('(\d+)\.(\d+)\.(\d+)\.(\d+)')
    matches = regex.match(gameVersion)
    return int(matches.group(1))


def separateMatchID(matchId: str):
    # TODO: implement regex
    pass

import re


def get_season(gameVersion: str) -> int:
    regex = re.compile('(\d+)\.(\d+)\.(\d+)\.(\d+)')
    matches = regex.match(gameVersion)
    return matches.group(1)

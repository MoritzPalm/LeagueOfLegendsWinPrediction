import re


def get_season(gameVersion: str) -> int:
    regex = re.compile('(\d+)\.(\d+)\.(\d+)\.(\d+)')
    matches = regex.match(gameVersion)
    print(matches)
    return matches[0]

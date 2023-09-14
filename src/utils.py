import logging
import re
import pandas as pd
from typing import Match


def parse_game_version(gameVersion: str) -> Match[str]:
    """
    this function parses the gameVersion string to extract season and patch information
    :param gameVersion: game version string with format dd.dd.ddd.ddd where d is any digit
    :return: Match object containing 4 match groups: 1 is season, 2 is patch number
    """
    regex = re.compile('(\d+)\.(\d+)\.(\d+)\.(\d+)')
    matches = regex.match(gameVersion)
    return matches


def get_season(gameVersion: str) -> int:
    matches = parse_game_version(gameVersion)
    return int(matches.group(1))


def get_patch(gameVersion: str) -> int:
    matches = parse_game_version(gameVersion)
    return int(matches.group(2))


def separateMatchID(matchId: str) -> tuple[str, int]:
    regex = re.compile("(.+)_(\d+)")
    matches = regex.match(matchId)
    platformId = matches.group(1)
    gameId = int(matches.group(2))
    return platformId, gameId


def clean_champion_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Win rate'] = df['Win rate'].str.strip('%')
    df['Pick Rate'] = df['Pick Rate'].str.strip('%')
    df['Ban Rate'] = df['Ban Rate'].str.strip('%')
    df['Matches'] = df['Matches'].str.replace(',', '').astype(int)
    return df


def is_valid_match(match_info: dict) -> bool:
    logging.debug(f"validating match info")
    if match_info['gameDuration'] < 960:  # 16 min = 960 sec
        logging.warning(f"match is too short: match length was {match_info['gameDuration']}s, more than 960s expected")
        return False
    if match_info['queueId'] != 420:
        # queue ID for Ranked 5v5 solo, see: https://static.developer.riotgames.com/docs/lol/queues.json
        logging.warning(f"match has wrong queue: queue was {match_info['queue']}, 420 expected")
        return False
    if match_info['mapId'] not in [1, 2, 11]:
        # map ids for summoners rift, see https://static.developer.riotgames.com/docs/lol/maps.json
        logging.warning(f"match was played on wrong map: played on map {match_info['mapId']}, 1, 2 or 11 expected")
        return False
    return True

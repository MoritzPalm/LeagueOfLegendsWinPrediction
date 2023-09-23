import logging
import math
import re
import pandas as pd
from typing import Match


def parse_game_version(gameVersion: str) -> Match[str]:
    """
    this function parses the gameVersion string to extract season and patch information
    :param gameVersion: game version string with format dd.dd.ddd.ddd where d is any digit
    :return: Match object containing 4 match groups: 1 is season, 2 is patch number
    """
    regex = re.compile(r'(\d+)\.(\d+)\.(\d+)\.(\d+)')
    matches = regex.match(gameVersion)
    return matches


def get_season(gameVersion: str) -> int:
    matches = parse_game_version(gameVersion)
    return int(matches.group(1))


def get_patch(gameVersion: str) -> int:
    matches = parse_game_version(gameVersion)
    return int(matches.group(2))


def separateMatchID(matchId: str) -> tuple[str, int]:
    regex = re.compile(r'(.+)_(\d+)')
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


def clean_summoner_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleans summoner data, assumes that dataframe contains at least one row
    :param df: dataframe containing at least one row with columns ['Champion', 'WinsLoses', 'Winrate', 'KDA',
    'KillsDeathsAssists', 'LP', 'MaxKills', 'MaxDeaths', 'CS', 'Damage', 'Gold']
    :return: cleaned dataframe
    """
    if df.shape[0] == 1:
        df_winsloses = df['WinsLoses'].str.extract(r'(\d+)W (\d+)L')
        df['Winrate'] = df['Winrate'].str.strip('%').astype(float, errors='ignore')
        df_killsdeathsassists = df['KillsDeathsAssists'].str.extract(
            r'(\d+.\d+)\/(\d+.\d+)\/(\d+.\d+)')  # regex matching 5.2, 4.0 and 5.1 from string "5.2/4.0/5.1"
        df['LP'] = df['LP'].str.strip('LP').astype(int, errors='ignore')
    else:
        df_winsloses = df['WinsLoses'].squeeze(axis=0).str.extract(r'(\d+)W (\d+)L')  # regex matching 12 and 5 from string "12W 5L"
        df['Winrate'] = df['Winrate'].str.strip('%').astype(float, errors='ignore')
        df_killsdeathsassists = df['KillsDeathsAssists'].squeeze().str.extract(
            r'(\d+.\d+)\/(\d+.\d+)\/(\d+.\d+)')  # regex matching 5.2, 4.0 and 5.1 from string "5.2/4.0/5.1"
        df['LP'] = df['LP'].str.strip('LP').astype(int, errors='ignore')
    df['wins'] = df_winsloses[0].astype(int, errors='ignore')
    df['loses'] = df_winsloses[1].astype(int, errors='ignore')
    df.loc[df['KDA'] == 'Perfect', 'KDA'] = math.inf   # inf means that perfect kda is achieved (0 deaths and >0 kills)
    df['KDA'] = df['KDA'].astype(float, errors='ignore')
    df['kills'] = df_killsdeathsassists[0].astype(float, errors='ignore')
    df['deaths'] = df_killsdeathsassists[1].astype(float, errors='ignore')
    df['assists'] = df_killsdeathsassists[0].astype(float, errors='ignore')
    df['MaxKills'] = df['MaxKills'].astype(int, errors='ignore')
    df['MaxDeaths'] = df['MaxDeaths'].astype(int, errors='ignore')
    df['CS'] = df['CS'].astype(float, errors='ignore')
    df['Damage'] = df['Damage'].astype(float, errors='ignore')
    df['Gold'] = df['Gold'].astype(float, errors='ignore')
    return df

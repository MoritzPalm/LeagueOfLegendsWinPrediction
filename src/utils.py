import datetime
import logging
import math
import re
from typing import Match

import pandas as pd


def parse_game_version(gameVersion: str) -> Match[str]:
    """
    this function parses the gameVersion string to extract season and patch information
    :param gameVersion: game version string with format dd.dd.ddd.ddd where d is any digit
    :return: Match object containing 4 match groups: 1 is season, 2 is patch number
    """
    regex = re.compile(r"(\d+)\.(\d+)\.(\d+)\.(\d+)")
    matches = regex.match(gameVersion)
    return matches


def get_season(gameVersion: str) -> int:
    matches = parse_game_version(gameVersion)
    return int(matches.group(1))


def get_patch(gameVersion: str) -> int:
    matches = parse_game_version(gameVersion)
    return int(matches.group(2))


def separateMatchID(matchId: str) -> tuple[str, int]:
    regex = re.compile(r"(.+)_(\d+)")
    matches = regex.match(matchId)
    platformId = matches.group(1)
    gameId = int(matches.group(2))
    return platformId, gameId


def clean_champion_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Win rate"] = df["Win rate"].str.strip("%")
    df["Pick Rate"] = df["Pick Rate"].str.strip("%")
    df["Ban Rate"] = df["Ban Rate"].str.strip("%")
    df["Matches"] = df["Matches"].str.replace(",", "").astype(int)
    return df


def is_valid_match(match_info: dict) -> bool:
    logging.debug("validating match info")
    if match_info["gameDuration"] < 960:  # 16 min = 960 sec
        logging.warning(
            f"match is too short: match length was {match_info['gameDuration']}s, more than 960s expected"
        )
        return False
    if match_info["queueId"] != 420:
        # queue ID for Ranked 5v5 solo, see: https://static.developer.riotgames.com/docs/lol/queues.json
        logging.warning(
            f"match has wrong queue: queue was {match_info['queue']}, 420 expected"
        )
        return False
    if match_info["mapId"] not in [1, 2, 11]:
        # map ids for summoners rift, see https://static.developer.riotgames.com/docs/lol/maps.json
        logging.warning(
            f"match was played on wrong map: played on map {match_info['mapId']}, 1, 2 or 11 expected"
        )
        return False
    return True


def clean_summoner_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleans summoner data, assumes that dataframe contains at least one row
    :param df: dataframe containing at least one row with columns ['Champion', 'WinsLoses', 'Winrate', 'KDA',
    'KillsDeathsAssists', 'LP', 'MaxKills', 'MaxDeaths', 'CS', 'Damage', 'Gold']
    :return: cleaned dataframe
    """
    df["winRate"] = df["winRate"].astype(float, errors="ignore")
    df["lp"] = df["lp"].astype(int, errors="ignore")
    df["wins"] = df["winsLoses"].astype(int, errors="ignore")
    df.loc[
        df["kda"] == "Perfect", "KDA"
    ] = math.inf  # inf means that perfect kda is achieved (0 deaths and >0 kills)
    df["kda"] = df["KDA"].astype(float, errors="ignore")
    df["kills"] = df["kills"].astype(float, errors="ignore")
    df["deaths"] = df["deaths"].astype(float, errors="ignore")
    df["assists"] = df["assists"].astype(float, errors="ignore")
    df["maxKills"] = df["maxKills"].astype(int, errors="ignore")
    df["cs"] = df["cs"].astype(float, errors="ignore")
    df["damage"] = df["damage"].astype(float, errors="ignore")
    df["gold"] = df["gold"].astype(float, errors="ignore")
    return df


def clean_champion_name(name: str) -> str:
    """
    Removes special characters and spaces from existing champion name
    :param name:
    :return:
    """
    cleaned = re.sub(r"[^\w\s]", "", name).replace(" ", "").lower()
    if cleaned == "wukong":
        cleaned = "monkeyking"
    if cleaned == "nunuwillump":
        cleaned = "nunu"
    if cleaned == "renataglasc":
        cleaned = "renata"
    return cleaned


def convert_patchNumber_time(season: int, patch: int) -> tuple[int, int]:
    """
    Converts patch number to Unix time
    numbers for datetime are from
    https://support-leagueoflegends.riotgames.com/hc/en-us/articles/360018987893-Patch-Schedule-League-of-Legends
    this function is returning the absolute bounds for the patch, so using this will result is some matches from
    other patches due to timezone issues. this is preferable to the alternative where some matches would be missed
    :param season: season number
    :param patch: patch number
    :return: Unix timestamp start, Unix timestamp end
    """
    if season == 13:
        if patch == 20:
            start_day = int(datetime.datetime(2023, 10, 11).timestamp())
            end_day = int(datetime.datetime(2023, 10, 25).timestamp())
        elif patch == 21:
            start_day = int(datetime.datetime(2023, 10, 25).timestamp())
            end_day = int(datetime.datetime(2023, 11, 8).timestamp())
        else:
            raise NotImplementedError(f"patch {season}.{patch} not implemented")
    else:
        raise NotImplementedError(f"season {season} not implemented")
    return start_day, end_day


def get_teamId_from_participantIds(participantIds: list[int]) -> int:
    """
    Returns the teamId of the team that the participantIds belong to
    :param participantIds: list of participantIds
    :return: teamId
    """
    team = set()

    for id in participantIds:
        if id > 10:
            raise ValueError(f"participantId {id} is invalid")
        if id <= 5:
            team.add(0)
        else:
            team.add(1)

    if len(team) != 1:
        raise ValueError(f"participantIds {participantIds} belong to both teams")

    return team.pop()

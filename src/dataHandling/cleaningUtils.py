from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd


def drop_missing(df: pd.DataFrame, thresh: int = 0) -> pd.DataFrame:
    """
    drops rows with missing values
    :param df: pd.DataFrame
    :param thresh: how many non-NaN values a row needs to have to not be dropped, default is len(df.columns) - 1
    :return: None
    """
    if thresh == 0:
        thresh = len(df.columns) - 1
    len_before = len(df)
    df_new = df.dropna(axis=0)
    print(f'dropped {len_before - len(df_new)} rows')
    return df_new


def replace_missing(df: pd.DataFrame, fill_value: int = -1) -> pd.DataFrame:
    """
    replaces missing values with fill_value
    :param df: pd.DataFrame
    :param fill_value: value to fill missing values with
    :return: None
    """
    df_new = df.fillna(fill_value)
    return df_new


def get_winning_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds a column 'label' to the dataframe where 0 = team1 won, 1 = team2 won
    :param df: pd.Dataframe with column 'participant1_win'
    :return: None
    """
    df['label'] = np.where(df['participant1_win'], 0, 1)  # 0 = team1 won, 1 = team2 won
    return df


def drop_wrong_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    asserts that all rows meet certain criteria and drops any row which does not meet one or more criteria
    :param df: pd.DataFrame
    :return: None
    """
    len_before = len(df)
    print(f'found {len(df)} rows')
    df.drop(df[df['mapId'] != 11].index, inplace=True)
    print('dropped wrong mapId')
    df.drop(df[df['queueId'] != 420].index, inplace=True)
    print('dropped wrong queueId')
    df.drop(df[df['gameDuration'] < 900].index, inplace=True)
    print('dropped wrong gameDuration')
    df.drop(df[df['seasonId'] != 13].index, inplace=True)
    print('dropped wrong seasonId')
    df.drop(df[df['gameVersion'] != df['gameVersion'][0]].index, inplace=True)
    print('dropped wrong gameVersion')
    df.drop(df[df['patch'] != df['patch'][0]].index, inplace=True)
    print('dropped wrong patch')
    print(f'dropped {len_before - len(df)} wrong rows')
    return df


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """
    drops columns which are irrelevant for the model (mostly ids)
    :param df: pd.DataFrame
    :return: None
    """
    irrelevant_cols = []
    general_irrelevant = ['gameDuration', 'gameCreation', 'gameVersion', 'mapId', 'queueId', 'patch', 'seasonId',
                          'platformId']
    irrelevant_cols.extend(general_irrelevant)
    participant_irrelevant = ['win', 'veteran', 'inactive', 'freshBlood', 'hotStreak',
                              'champion_championPointsSinceLastLevel',
                              'champion_tokensEarned',
                              'champion_infoAttack', 'champion_infoDefense', 'champion_infoMagic',
                              'champion_infoDifficulty', 'champion_matches', 'teamId', 'champion_championLevel',
                              'champion_kills', 'champion_deaths', 'champion_assists', 'champion_maxKills',
                              'champion_cs', 'champion_damage', 'champion_gold', 'champion_championNumber',
                              'champion_pick_rate', 'champion_ban_rate', 'leaguePoints']
    for i in range(1, 11):
        for col in participant_irrelevant:
            irrelevant_cols.append(f'participant{i}_{col}')
    df_new = df.drop(columns=irrelevant_cols)
    return df_new


class Tier(Enum):
    """
    Enum for the different tiers
    """
    IRON = 0
    BRONZE = 1
    SILVER = 2
    GOLD = 3
    PLATINUM = 4
    EMERALD = 5
    DIAMOND = 6
    MASTER = 7
    GRANDMASTER = 8
    CHALLENGER = 9


def format_rank(tier: str, rank: str) -> str:
    """
    formats the rank from two str into a float
    :param tier: major rank group, e.g. MASTER (see Tier class)
    :param rank: minor rank group, e.g. I, II, III, IV
    :return: formatted rank, e.g. 7.1
    """
    return f'{tier}.{rank}'


def fix_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    converts the rank from two str columns into one float column and drops the rank column
    :param df: pd.DataFrame with columns 'participant{i}_tier' and 'participant{i}_rank' where i is a number from 1 to 10
    :return: None
    """
    for i in range(1, 11):
        df[f'participant{i}_tier'] = df[f'participant{i}_tier'].apply(lambda x: Tier[x].value)
        df.loc[:, f'participant{i}_tier'] = df.apply(
            lambda x: format_rank(x[f'participant{i}_tier'], x[f'participant{i}_rank']), axis=1)
        df[f'participant{i}_tier'] = df[f'participant{i}_tier'].astype(float)
        df.drop(columns=[f'participant{i}_rank'], inplace=True)
    return df


def calc_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates the winrate for each participant and drops the wins and losses columns
    :param df: pd.DataFrame with columns 'participant{i}_wins' and 'participant{i}_losses' where i is a number from 1 to
     10
    :return: None
    """
    for i in range(1, 11):
        df[f'participant{i}_winrate'] = df[f'participant{i}_wins'] / (
                df[f'participant{i}_wins'] + df[f'participant{i}_losses'])
        df.drop(columns=[f'participant{i}_wins', f'participant{i}_losses'], inplace=True)
    return df


def fix_teamId(df: pd.DataFrame) -> pd.DataFrame:
    """
    converts the teamId from 100/200 to 0/1
    :param df: pd.DataFrame with columns 'participant{i}_teamId' where i is a number from 1 to 10
    :return: None
    """
    for i in range(1, 11):
        df[f'participant{i}_teamId'] = df[f'participant{i}_teamId'] // 100 - 1
    return df


def convert_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """
    converts boolean columns to int columns
    :param df: pd.DataFrame
    :return: None
    """
    df_new = df.replace({True: 1, False: 0})
    return df_new


def convert_lastPlayTime(df: pd.DataFrame) -> pd.DataFrame:
    """
    converts the lastPlayTime column to the time since last playtime in seconds
    :param df: pd.DataFrame with columns 'participant{i}_champion_lastPlayTime' where i is a number from 1 to 10
    :return: None
    """
    for i in range(1, 11):
        df[f'participant{i}_champion_lastPlayTime'] = df[f'participant{i}_champion_lastPlayTime'].apply(
            lambda x: int((datetime.now() - datetime.fromtimestamp(x / 1000)).total_seconds()) if x is not np.nan
            else 0)
    return df


class championTier(Enum):
    """
    Enum for the different champion tiers where SS is S+
    """
    D = 0
    C = 1
    B = 2
    A = 3
    S = 4
    SS = 5


def convert_championTier(df: pd.DataFrame) -> pd.DataFrame:
    """
    converts the champion tier column to a int column
    :param df: pd.DataFrame with columns 'participant{i}_champion_tier' where i is a number from 1 to 10
    :return: DataFrame with changed column
    """
    for i in range(1, 11):
        df[f'participant{i}_champion_tier'] = (df[f'participant{i}_champion_tier']
                                               .apply(lambda x: championTier[x]
                                                      .value if x != "S+" else 5))
    return df


def drop_wrong_teamIds(df: pd.DataFrame) -> pd.DataFrame:
    """
    drops rows where the teamId is not 0 for the first 5 participants and not 1 for the last 5 participants
    :param df:
    :return:
    """
    len_before = len(df)
    for i in range(1, 6):
        df.drop(df[df[f'participant{i}_teamId'] != 0].index, inplace=True)
    for i in range(6, 11):
        df.drop(df[df[f'participant{i}_teamId'] != 1].index, inplace=True)
    print(f'dropped {len_before - len(df)} rows')
    return df

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
    df_new = df.dropna(axis=0, thresh=thresh)
    print(f'dropped {len_before - len(df)} rows')
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
    df.drop(df[df['mapId'] != 11].index, inplace=True)
    df.drop(df[df['queueId'] != 420].index, inplace=True)
    df.drop(df[df['gameDuration'] < 900].index, inplace=True)
    df.drop(df[df['platformId'] != 'EUW1'].index, inplace=True)
    df.drop(df[df['seasonId'] != 13].index, inplace=True)
    df.drop(df[df['gameVersion'] != df['gameVersion'][0]].index, inplace=True)
    df.drop(df[df['patch'] != df['patch'][0]].index, inplace=True)
    print(f'dropped {len_before - len(df)} rows')
    return df


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """
    drops columns which are irrelevant for the model (mostly ids)
    :param df: pd.DataFrame
    :return: None
    """
    irrelevant_cols = ['gameDuration', 'gameCreation', 'gameVersion', 'mapId', 'queueId', 'patch', 'seasonId',
                       'platformId']
    for i in range(1, 11):
        irrelevant_cols.append(f'participant{i}_win')
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
    for i in range(2, 11):
        df.loc[f'participant{i}_tier'] = df.loc[f'participant{i}_tier'].apply(lambda x: Tier[x].value)
        df.loc[:, f'participant{i}_tier'] = df.apply(
            lambda x: format_rank(x[f'participant{i}_tier'], x[f'participant{i}_rank']), axis=1)
        df.loc[f'participant{i}_tier'] = df.loc[f'participant{i}_tier'].astype(float)
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
        df.loc[f'participant{i}_teamId'] = df[f'participant{i}_teamId'] // 100 - 1
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
        df.loc[f'participant{i}_champion_lastPlayTime'] = df.loc[f'participant{i}_champion_lastPlayTime'].apply(
            lambda x: int((datetime.now() - datetime.fromtimestamp(x / 1000)).total_seconds()))
    return df

import re
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
                              'champion_pick_rate', 'champion_ban_rate', 'leaguePoints', 'champion_tier']
    participant_irrelevant = ['win', 'lp']
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
        df[f'participant{i}_lastPlayTime'] = df[f'participant{i}_lastPlayTime'].apply(
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
    print(f'dropped {len_before - len(df)} rows because of wrong teamIds')
    return df


def drop_wrong_wins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only rows where the first 5 participants have the same value (True/False)
    and the last 5 participants have the opposite value.

    :param df: DataFrame with columns named "participant<x>_win" where x is a number from 1 to 10
    :return: Filtered DataFrame
    """
    len_before = len(df)
    # Columns for first 5 participants and last 5 participants
    first_five_cols = [f'participant{i}_win' for i in range(1, 6)]
    last_five_cols = [f'participant{i}_win' for i in range(6, 11)]

    # Check the condition for each row
    valid_rows = df.apply(
        lambda row: (row[first_five_cols].nunique() == 1) and
                    (row[last_five_cols].nunique() == 1) and
                    (row[first_five_cols[0]] != row[last_five_cols[0]]),
        axis=1
    )
    print(f'dropped {len_before - len(df[valid_rows])} rows because of wrong wins')
    # Filter the DataFrame
    return df[valid_rows]


def average_over_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    averages all participant columns into one column for each team per category
    :param df: pd.DataFrame
    :return: DataFrame with averaged columns
    """
    raise NotImplementedError


def merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges columns of the form participant<x>_<col> into two columns per category, one for each team.
    Uses participant<x>_team columns to determine the team of each participant.
    Drops categorical columns that are not to be merged.
    :param df: DataFrame containing the columns to be merged.
    :return: Dictionary with keys as categories and values as a DataFrame with two columns per category, one for each team.
    """
    cols = df.columns.tolist()
    merged_columns = {}
    # cols that are not to be merged as they are categorical, so averaging them does not make sense
    cols_left = ['teamId', 'champion_championNumber']
    for col in cols:
        matches = re.search(r"participant(\d+)_(\w+)", col)
        if matches and matches.group(2) not in cols_left:
            participant_number = matches.group(1)
            col_type = matches.group(2)

            # Determine the team of the participant
            team_col = f'participant{participant_number}_teamId'
            team = df[team_col].iloc[0] if team_col in df.columns else 'unknown'

            if col_type not in merged_columns:
                merged_columns[col_type] = {'team1': [], 'team2': []}

            if team == 0:
                merged_columns[col_type]['team1'].append(df[col])
            elif team == 1:
                merged_columns[col_type]['team2'].append(df[col])

    merged_series = {}
    for col_type, teams_data in merged_columns.items():
        for team, data_list in teams_data.items():
            if data_list:  # Only proceed if there are columns to merge
                merged_series[f"{col_type}_{team}"] = pd.concat(data_list, axis=1).mean(axis=1)

    return pd.DataFrame(merged_series)

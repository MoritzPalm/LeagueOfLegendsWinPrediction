import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops irrelevant columns from the DataFrame
    irrelevant columns are: participantId, abilityHaste, armorPen, bonusArmorPenPercent, bonusMagicPenPercent,
    cooldownReduction, physicalVamp, spellVamp, goldPerSecond
    :param df: DataFrame to drop columns from
    :return: DataFrame with dropped columns
    """
    irrelevant_cols = ['participantId', 'abilityHaste', 'armorPen', 'bonusArmorPenPercent', 'bonusMagicPenPercent',
                       'cooldownReduction', 'physicalVamp', 'spellVamp', 'goldPerSecond']
    for i in range(1, 11):
        for col in irrelevant_cols:
            col = f'participant{i}_{col}'
            df.drop(col, axis=1, inplace=True)
    return df


def make_label_last_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moves the label column to the last column of the DataFrame
    :param df: DataFrame to move label column in
    :return: DataFrame with label column moved
    """
    cols = [col for col in df.columns if col != 'winning_team'] + ['winning_team']
    df = df[cols]
    return df


def prune_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prunes the timeline to not include events after 16 minutes (960000 milliseconds)
    :param df: DataFrame to prune
    :return: Pruned DataFrame
    """
    len_before = len(df)
    df_new = df.drop(df.query("timestamp > 960000").index)
    print(f'Pruned {len_before - len(df_new)} rows')
    return df_new


def drop_short_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets matches that are shorter than 16 minutes
    :param df: DataFrame to drop matches from
    :return: DataFrame with dropped matches
    """
    len_before = len(df)
    matchIds = df.index.get_level_values('matchId').unique()
    short_matches = []
    for matchId in matchIds:
        match_df = df.loc[matchId]
        if match_df.iloc[-1]['timestamp'] < 900000:
            short_matches.append(matchId)
    df_new = df.drop(short_matches, level='matchId')
    print(f'Dropped {len_before - len(df_new)} rows or {len(short_matches)} matches')
    return df_new


def train_test_split(df: pd.DataFrame, test_size: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the DataFrame into a training and test set
    :param df: DataFrame to split
    :param test_size: Size of the test set as a fraction of the total set
    :return: train_df, train_labels, test_df, test_labels
    """
    matchIds = df.index.get_level_values('matchId').unique()
    test_length = int(len(matchIds) * test_size)
    test_matchIds = matchIds[:test_length]
    train_matchIds = matchIds[test_length:]
    test_df = df.loc[test_matchIds]
    train_df = df.loc[train_matchIds]
    train_labels = train_df.pop('winning_team')
    test_labels = test_df.pop('winning_team')
    return train_df, train_labels, test_df, test_labels


def test_match_length(df: pd.DataFrame) -> bool:
    """
    tests if all matchIds have exactly 16 timestamps
    :param df: DataFrame to test
    :return: True if all matches have 16 timestamps, False otherwise
    """
    matchIds = df.index.get_level_values('matchId').unique()
    found_short = False
    print(f'Found {len(matchIds)} matches')
    for matchId in matchIds:
        match_df = df.loc[matchId]
        if len(match_df) != 16:
            print(f'Match {matchId} has {len(match_df)} timestamps')
            found_short = True
    if not found_short:
        print('All matches have 16 timestamps')
        return True
    else:
        return False


def cleanTimelineDataset():
    """
    Cleans the timeline dataset and saves it to the data/processed folder
    :return: None
    """
    with open('../data/raw/timelines.pkl', 'rb') as f:
        df = pickle.load(f)
        print(df.shape)
    df = df.sort_values(by=['matchId', 'timestamp'])
    df = drop_irrelevant_columns(df)
    df = make_label_last_col(df)
    df = prune_timeline(df)
    df = drop_short_matches(df)
    if not test_match_length(df):
        raise ValueError('Timeline length of at least one match is not 16')
    train_df, train_labels, test_df, test_labels = train_test_split(df, 0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df)
    X_test = scaler.transform(test_df)
    train_df = np.append(X_train, np.expand_dims(train_labels, axis=1), axis=1)
    X_train = np.append(X_test, np.expand_dims(test_labels, axis=1), axis=1)
    np.save('../data/processed/train_timeline', X_train)
    np.save('../data/processed/test_timeline', X_test)

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


def train_val_test_split(df: pd.DataFrame, test_size: float = 0.1, val_size=0.1) -> (pd.DataFrame, pd.DataFrame,
                                                                                     pd.DataFrame,
                                                                                     pd.DataFrame):
    """
    Splits the DataFrame into a training, validation and test set, where the validation and test sets have the same
    lengths
    :param df: DataFrame to split
    :param test_size: Size of the test set as a fraction of the total set
    :return: train_df, train_labels, test_df, test_labels
    """
    assert 1 - test_size * 2 > 0
    matchIds = df.index.get_level_values('matchId').unique()
    test_length = int(len(matchIds) * test_size)
    val_length = int(len(matchIds) * val_size)
    test_matchIds = matchIds[:test_length]
    val_matchIds = matchIds[test_length:test_length + val_length]
    train_matchIds = matchIds[test_length + val_length:]
    test_df = df.loc[test_matchIds]
    train_df = df.loc[train_matchIds]
    val_df = df.loc[val_matchIds]
    train_labels = train_df.pop('winning_team')
    test_labels = test_df.pop('winning_team')
    val_labels = val_df.pop('winning_team')
    check_correct_split(df, train_df, test_df, val_df)
    return train_df, train_labels, test_df, test_labels, val_df, val_labels


def check_correct_split(df: pd.DataFrame, train_df, test_df, val_df):
    """
    Checks if the split was correct and no data leakage occurred
    :param df:
    :param train_df:
    :param test_df:
    :param val_df:
    :return:
    """
    train_matchIds = train_df.index.get_level_values('matchId').unique()
    test_matchIds = test_df.index.get_level_values('matchId').unique()
    val_matchIds = val_df.index.get_level_values('matchId').unique()
    assert len(train_matchIds) + len(test_matchIds) + len(val_matchIds) == len(
        df.index.get_level_values('matchId').unique())
    assert len(set(train_matchIds).intersection(set(test_matchIds))) == 0
    assert len(set(train_matchIds).intersection(set(val_matchIds))) == 0
    assert len(set(test_matchIds).intersection(set(val_matchIds))) == 0
    all_matchIds = set(train_matchIds).union(test_matchIds).union(val_matchIds)  # this removes duplicate
    # matchIds
    assert len(all_matchIds) == len(df.index.get_level_values('matchId').unique())


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
    with open('data/raw/timeline_dataset.pkl', 'rb') as f:
        df = pickle.load(f)
        print(df.shape)
    df = df.sort_values(by=['matchId', 'timestamp'])
    df = drop_irrelevant_columns(df)
    df = make_label_last_col(df)
    df = prune_timeline(df)
    df = drop_short_matches(df)
    if not test_match_length(df):
        raise ValueError('Timeline length of at least one match is not 16')
    train_df, train_labels, test_df, test_labels, val_df, val_labels = train_val_test_split(df, 0.0, 0.1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df)
    X_test = scaler.transform(test_df)
    X_val = scaler.transform(val_df)
    X_train = np.append(X_train, np.expand_dims(train_labels, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(test_labels, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(val_labels, axis=1), axis=1)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'X_val shape: {X_val.shape}')
    np.save('data/processed/train_timeline', X_train)
    np.save('data/processed/test_timeline', X_test)
    np.save('data/processed/val_timeline', X_val)


if __name__ == '__main__':
    cleanTimelineDataset()

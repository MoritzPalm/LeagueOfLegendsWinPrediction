import glob
import logging
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
    irrelevant_cols = ["participantId", "abilityHaste", "armorPen", "bonusArmorPenPercent", "bonusMagicPenPercent",
                       "cooldownReduction", "physicalVamp", "spellVamp", "goldPerSecond"]
    for i in range(1, 11):
        for col in irrelevant_cols:
            col = f"participant{i}_{col}"
            df.drop(col, axis=1, inplace=True)
    return df


def make_label_last_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moves the label column to the last column of the DataFrame
    :param df: DataFrame to move label column in
    :return: DataFrame with label column moved
    """
    cols = [col for col in df.columns if col != "winning_team"] + ["winning_team"]
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
    print(f"Pruned {len_before - len(df_new)} rows")
    return df_new


def drop_short_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets matches that are shorter than 16 minutes
    :param df: DataFrame to drop matches from
    :return: DataFrame with dropped matches
    """
    len_before = len(df)
    matchIds = df.index.get_level_values("matchId").unique()
    short_matches = []
    for matchId in matchIds:
        match_df = df.loc[matchId]
        if match_df.iloc[-1]["timestamp"] < 900000:
            short_matches.append(matchId)
    df_new = df.drop(short_matches, level="matchId")
    print(f"Dropped {len_before - len(df_new)} rows or {len(short_matches)} matches")
    return df_new


def train_val_test_split(df: pd.DataFrame, test_size: float | int = 0.1,
                         val_size: float | int = 0.1) -> (pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame):
    """
    Splits the DataFrame into a training, validation and test set, where the validation and test sets have the same
    lengths
    :param val_size:
    :param df: DataFrame to split
    :param test_size: Size of the test set as a fraction of the total set
    :return: train_df, train_labels, test_df, test_labels
    """
    matchIds = df.index.get_level_values("matchId").unique()
    if isinstance(test_size, int):
        test_length = test_size
        val_length = val_size
    elif isinstance(test_size, float):
        test_length = int(len(matchIds) * test_size)
        val_length = int(len(matchIds) * val_size)
    else:
        raise ValueError("test_size must be either int or float")
    # assert 1 - test_size * 2 > 0
    test_matchIds = matchIds[:test_length]
    val_matchIds = matchIds[test_length:test_length + val_length]
    train_matchIds = matchIds[test_length + val_length:]
    test_df = df.loc[test_matchIds]
    train_df = df.loc[train_matchIds]
    val_df = df.loc[val_matchIds]
    train_labels = train_df.pop("winning_team")
    test_labels = test_df.pop("winning_team")
    val_labels = val_df.pop("winning_team")
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
    train_matchIds = train_df.index.get_level_values("matchId").unique()
    test_matchIds = test_df.index.get_level_values("matchId").unique()
    val_matchIds = val_df.index.get_level_values("matchId").unique()
    assert len(train_matchIds) + len(test_matchIds) + len(val_matchIds) == len(
        df.index.get_level_values("matchId").unique())
    assert len(set(train_matchIds).intersection(set(test_matchIds))) == 0
    assert len(set(train_matchIds).intersection(set(val_matchIds))) == 0
    assert len(set(test_matchIds).intersection(set(val_matchIds))) == 0
    all_matchIds = set(train_matchIds).union(test_matchIds).union(val_matchIds)  # this removes duplicate
    # matchIds
    assert len(all_matchIds) == len(df.index.get_level_values("matchId").unique())


def test_match_length(df: pd.DataFrame) -> bool:
    """
    tests if all matchIds have exactly 16 timestamps
    :param df: DataFrame to test
    :return: True if all matches have 16 timestamps, False otherwise
    """
    matchIds = df.index.get_level_values("matchId").unique()
    found_short = False
    print(f"Found {len(matchIds)} matches")
    for matchId in matchIds:
        match_df = df.loc[matchId]
        if len(match_df) != 16:
            print(f"Match {matchId} has {len(match_df)} timestamps")
            found_short = True
    if not found_short:
        print("All matches have 16 timestamps")
        return True
    else:
        return False


def average_over_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages the values of the columns over the two teams
    :param df: DataFrame where the columns are named participant{i}_{type}
    :return:
    """
    # Columns that should be left as they are
    columns_left = ["timestamp", "winning_team"]
    # Extract unique types from the column names
    types = set(col.split("_")[-1] for col in df.columns)

    # Create a new DataFrame for the results
    result_df = pd.DataFrame()

    for type in types:
        # Columns for participants 1-5
        cols_1_to_5 = [f"participant{i}_{type}" for i in range(1, 6) if f"participant{i}_{type}" in df.columns]
        # Columns for participants 6-10
        cols_6_to_10 = [f"participant{i}_{type}" for i in range(6, 11) if f"participant{i}_{type}" in df.columns]

        # Calculate averages and add to the result DataFrame
        if cols_1_to_5:
            result_df[f"team0_{type}"] = df[cols_1_to_5].mean(axis=1)
        if cols_6_to_10:
            result_df[f"team1_{type}"] = df[cols_6_to_10].mean(axis=1)

    # Add the remaining columns
    for col in columns_left:
        result_df[col] = df[col]

    return result_df


def drop_columns_not_including(df, substrings):
    """
    Drops columns that do not include any of the substrings
    :param df:
    :param substrings:
    :return:
    """
    # Keep only columns that contain any of the substrings
    relevant_columns = [col for col in df.columns if any(substring in col for substring in substrings)]

    # Drop columns that are not in relevant_columns
    return df[relevant_columns]


def cleanTimelineDataset(save=True, concatenated=False):
    """
    Cleans the timeline dataset and saves it to the data/processed folder
    :return: None
    """
    dir = "data/timeline_25_12_23"
    if not concatenated:
        df = pd.DataFrame()
        for f in glob.glob(f"{dir}/raw/*.pkl"):
            with open(f, "rb") as file:
                df_new = pickle.load(file)
            df = pd.concat([df, df_new], axis=0)
        df.to_pickle(f"{dir}/timeline_full_raw.pkl")  # this cannot be saved in the /raw/
        # dir as it would not stand out with the glob regex
        logging.info("Concatenated all files")
    else:
        with open(f"{dir}/timeline_full_raw.pkl", "rb") as f:
            df = pickle.load(f)
            print(df.shape)
    df = df.sort_values(by=["matchId", "timestamp"])
    df = drop_irrelevant_columns(df)
    df = make_label_last_col(df)
    df = prune_timeline(df)
    df = drop_short_matches(df)

    columns = df.columns

    if not test_match_length(df):
        raise ValueError("Timeline length of at least one match is not 16")
    train_df, train_labels, test_df, test_labels, val_df, val_labels = train_val_test_split(df, 1000, 1000)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df)
    X_test = scaler.transform(test_df)
    X_val = scaler.transform(val_df)

    X_train = np.append(X_train, np.expand_dims(train_labels, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(test_labels, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(val_labels, axis=1), axis=1)

    df_train = pd.DataFrame(X_train, columns=columns)
    df_test = pd.DataFrame(X_test, columns=columns)
    df_val = pd.DataFrame(X_val, columns=columns)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    if save:
        dir_full = f"{dir}/processed/full"
        np.save(f"{dir_full}/train_timeline", X_train)
        np.save(f"{dir_full}/test_timeline", X_test)
        np.save(f"{dir_full}/val_timeline", X_val)
        df_train.to_pickle(f"{dir_full}/train_timeline.pkl")
        df_test.to_pickle(f"{dir_full}/test_timeline.pkl")
        df_val.to_pickle(f"{dir_full}/val_timeline.pkl")

    # averaging over teams
    df = average_over_teams(df)
    columns = df.columns
    (train_avg_df, train_avg_labels, test_avg_df, test_avg_labels, val_avg_df, val_avg_labels) = (
        train_val_test_split(df, 1000, 1000))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_avg_df)
    X_test = scaler.transform(test_avg_df)
    X_val = scaler.transform(val_avg_df)

    X_train = np.append(X_train, np.expand_dims(train_avg_labels, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(test_avg_labels, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(val_avg_labels, axis=1), axis=1)

    df_train = pd.DataFrame(X_train, columns=columns)
    df_test = pd.DataFrame(X_test, columns=columns)
    df_val = pd.DataFrame(X_val, columns=columns)

    if save:
        dir_avg = f"{dir}/processed/avg"
        np.save(f"{dir_avg}/train_timeline", X_train)
        np.save(f"{dir_avg}/test_timeline", X_test)
        np.save(f"{dir_avg}/val_timeline", X_val)
        df_train.to_pickle(f"{dir_avg}/train_timeline.pkl")
        df_test.to_pickle(f"{dir_avg}/test_timeline.pkl")
        df_val.to_pickle(f"{dir_avg}/val_timeline.pkl")

    # averaged over teams only gold
    cols = ["totalGold", "winning_team"]
    df_gold = drop_columns_not_including(df, cols)
    df_gold_cols = df_gold.columns

    (train_gold_df, train_gold_labels, test_gold_df, test_gold_labels, val_gold_df, val_gold_labels) = (
        train_val_test_split(df_gold, 1000, 1000))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_gold_df)
    X_test = scaler.transform(test_gold_df)
    X_val = scaler.transform(val_gold_df)

    X_train = np.append(X_train, np.expand_dims(train_gold_labels, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(test_gold_labels, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(val_gold_labels, axis=1), axis=1)

    df_train = pd.DataFrame(X_train, columns=df_gold_cols)
    df_test = pd.DataFrame(X_test, columns=df_gold_cols)
    df_val = pd.DataFrame(X_val, columns=df_gold_cols)

    if save:
        dir_gold = f"{dir}/processed/gold"
        np.save(f"{dir_gold}/train_timeline", X_train)
        np.save(f"{dir_gold}/test_timeline", X_test)
        np.save(f"{dir_gold}/val_timeline", X_val)
        df_train.to_pickle(f"{dir_gold}/train_timeline.pkl")
        df_test.to_pickle(f"{dir_gold}/test_timeline.pkl")
        df_val.to_pickle(f"{dir_gold}/val_timeline.pkl")

    # averaged over teams manual selection
    cols = ["totalGold", "kills", "level", "totalDamageDone", "winning_team"]
    df_manual = drop_columns_not_including(df, cols)
    df_manual_cols = df_manual.columns

    (train_manual_df, train_manual_labels, test_manual_df, test_manual_labels, val_manual_df, val_manual_labels) = (
        train_val_test_split(df_manual, 1000, 1000))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_manual_df)
    X_test = scaler.transform(test_manual_df)
    X_val = scaler.transform(val_manual_df)

    X_train = np.append(X_train, np.expand_dims(train_manual_labels, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(test_manual_labels, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(val_manual_labels, axis=1), axis=1)

    df_train = pd.DataFrame(X_train, columns=df_manual_cols)
    df_test = pd.DataFrame(X_test, columns=df_manual_cols)
    df_val = pd.DataFrame(X_val, columns=df_manual_cols)

    if save:
        dir_manual = f"{dir}/processed/manual"
        np.save(f"{dir_manual}/train_timeline", X_train)
        np.save(f"{dir_manual}/test_timeline", X_test)
        np.save(f"{dir_manual}/val_timeline", X_val)
        df_train.to_pickle(f"{dir_manual}/train_timeline.pkl")
        df_test.to_pickle(f"{dir_manual}/test_timeline.pkl")
        df_val.to_pickle(f"{dir_manual}/val_timeline.pkl")


if __name__ == "__main__":
    cleanTimelineDataset(save=True, concatenated=True)

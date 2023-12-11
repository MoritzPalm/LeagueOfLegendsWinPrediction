import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.dataHandling.cleaningUtils as clean


def cleanStaticDataset(save: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Cleans the static dataset and saves it to data/processed
    :return: X_train, X_val, X_test
    """
    print("Cleaning static dataset...")
    # df = pd.DataFrame()
    # for f in glob.glob('data/static_05_12_23/raw/raw/match_*.pkl'):
    #     with open(f, 'rb') as file:
    #         df_new = pickle.load(file)
    #     df = pd.concat([df, df_new], axis=0)
    # df.to_pickle('data/static_05_12_23/raw/static_full.pkl')
    logging.info("Concatenated all files")
    df = pd.read_pickle('data/static_05_12_23/raw/static_full.pkl')
    df = clean.drop_wrong_data(df)
    df.reset_index(drop=True, inplace=True)
    df = clean.fix_rank(df)
    df = clean.calc_winrate(df)
    df = clean.fix_teamId(df)
    df = clean.convert_booleans(df)
    df = clean.convert_lastPlayTime(df)
    df = clean.convert_championTier(df)
    df = clean.get_winning_team(df)
    df = clean.drop_irrelevant(df)
    df = clean.drop_missing(df)
    df = clean.drop_wrong_teamIds(df)

    assert df.columns[-1] == 'label'

    print(f'Found {len(df)} rows after cleaning')
    print(f'number of rows with at least one nan in dataset: {df.isna().sum()}')

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df.iloc[:, -1],
                                                        test_size=0.1,
                                                        random_state=42,
                                                        shuffle=True, stratify=df.iloc[:, -1])
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=42,
                                                      shuffle=True, stratify=y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    # adding the label column back
    X_train = np.append(X_train, np.expand_dims(y_train, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(y_test, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(y_val, axis=1), axis=1)
    # converting to dataframes for easier visualization
    X_train = pd.DataFrame(X_train, columns=df.columns)
    X_test = pd.DataFrame(X_test, columns=df.columns)
    X_val = pd.DataFrame(X_val, columns=df.columns)

    if save:
        X_train.to_pickle('data/static_05_12_23/processed/train_static.pkl')
        X_test.to_pickle('data/static_05_12_23/processed/test_static.pkl')
        X_val.to_pickle('data/static_05_12_23/processed/val_static.pkl')
        np.save('data/static_05_12_23/processed/train_static.npy', X_train)
        np.save('data/static_05_12_23/processed/test_static.npy', X_test)
        np.save('data/static_05_12_23/processed/val_static.npy', X_val)

    # handling of averaged dataset

    # create columns containing the average of the team
    df_merged = pd.DataFrame(clean.merge_columns(df))

    # one hot encoding of championNumber
    categorical_columns = [f'participant{x}_champion_championNumber' for x in range(1, 11)]
    df_categorical = df.loc[:, categorical_columns]
    df_categorical_one_hot = pd.get_dummies(df_categorical.loc[:, 'participant1_champion_championNumber'],
                                            columns='participant1_champion_championNumber', prefix='championNumber')
    df_merged_ohc = pd.concat([df_merged, df_categorical_one_hot], axis=1)
    df_merged_ohc['label'] = df['label']

    assert df_merged_ohc.columns[-1] == 'label'

    (X_train_merged, X_test_merged,
     y_train_merged, y_test_merged) = train_test_split(df_merged_ohc.iloc[:, :-1],
                                                       df_merged_ohc.iloc[:, -1],
                                                       test_size=0.1,
                                                       random_state=42,
                                                       shuffle=True,
                                                       stratify=df_merged_ohc.iloc[:, -1])
    (X_train_merged, X_val_merged,
     y_train_merged, y_val_merged) = train_test_split(X_train_merged,
                                                      y_train_merged,
                                                      test_size=0.1,
                                                      random_state=42,
                                                      shuffle=True,
                                                      stratify=y_train_merged)

    numerical_columns = df_merged.columns.tolist()  # df_merged contains only averaged columns, categorical columns
    # are dropped
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numerical_columns)],
        remainder='passthrough')

    transformer = preprocessor.fit(X_train_merged)
    X_train_merged = transformer.transform(X_train_merged)
    X_test_merged = transformer.transform(X_test_merged)
    X_val_merged = transformer.transform(X_val_merged)
    X_train_merged = np.append(X_train_merged, np.expand_dims(y_train_merged, axis=1), axis=1)
    X_test_merged = np.append(X_test_merged, np.expand_dims(y_test_merged, axis=1), axis=1)
    X_val_merged = np.append(X_val_merged, np.expand_dims(y_val_merged, axis=1), axis=1)
    df_train_merged = pd.DataFrame(X_train_merged, columns=df_merged_ohc.columns)
    df_test_merged = pd.DataFrame(X_test_merged, columns=df_merged_ohc.columns)
    df_val_merged = pd.DataFrame(X_val_merged, columns=df_merged_ohc.columns)

    if save:
        df_train_merged.to_pickle('data/static_05_12_23/processed/train_static_merged.pkl')
        df_test_merged.to_pickle('data/static_05_12_23/processed/test_static_merged.pkl')
        df_val_merged.to_pickle('data/static_05_12_23/processed/val_static_merged.pkl')
        np.save('data/static_05_12_23/processed/train_static_merged.npy', X_train_merged)
        np.save('data/static_05_12_23/processed/test_static_merged.npy', X_test_merged)
        np.save('data/static_05_12_23/processed/val_static_merged.npy', X_val_merged)

    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'X_val shape: {X_val.shape}')
    print("Done cleaning static dataset")
    return X_train, X_val, X_test


if __name__ == '__main__':
    cleanStaticDataset(save=True)

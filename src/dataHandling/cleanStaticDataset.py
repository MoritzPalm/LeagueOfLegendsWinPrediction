import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.dataHandling.cleaningUtils as clean


def cleanStaticDataset(save: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Cleans the static dataset and saves it to data/processed
    :return: X_train, X_val, X_test
    """
    print("Cleaning static dataset...")
    with open("data/raw/static_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    print(f'Found {len(df)} rows')
    df = clean.drop_missing(df)
    print(f'Found {len(df)} rows after dropping missing values')
    df = clean.drop_wrong_data(df)
    df.reset_index(drop=True, inplace=True)
    df = clean.fix_rank(df)
    df = clean.calc_winrate(df)
    df = clean.fix_teamId(df)
    df = clean.convert_booleans(df)
    df = clean.convert_lastPlayTime(df)
    df = clean.convert_championTier(df)
    df = clean.get_winning_team(df)  # this has to be the last step where a column is inserted
    df = clean.drop_wrong_teamIds(df)
    df = clean.drop_irrelevant(df)
    assert df.columns[-1] == 'label'
    print(f'Found {len(df)} rows after cleaning')
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df.iloc[:, -1],
                                                        test_size=0.1,
                                                        random_state=42,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      random_state=42,
                                                      shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    X_train = np.append(X_train, np.expand_dims(y_train, axis=1), axis=1)
    X_test = np.append(X_test, np.expand_dims(y_test, axis=1), axis=1)
    X_val = np.append(X_val, np.expand_dims(y_val, axis=1), axis=1)
    if save:
        np.save('data/processed/train_static', X_train)
        np.save('data/processed/test_static', X_test)
        np.save('data/processed/val_static', X_val)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'X_val shape: {X_val.shape}')
    print("Done cleaning static dataset")
    return X_train, X_val, X_test


if __name__ == '__main__':
    cleanStaticDataset(save=True)

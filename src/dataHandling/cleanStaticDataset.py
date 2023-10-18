import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.dataHandling.cleaningUtils as clean

with open("data/raw/static_dataset.pkl", "rb") as f:
    df = pickle.load(f)

df = clean.drop_missing(df)
df = clean.get_winning_team(df)
df = clean.drop_wrong_data(df)
df = clean.drop_irrelevant(df)
df.reset_index(drop=True, inplace=True)
df = clean.fix_rank(df)
df = clean.calc_winrate(df)
df = clean.fix_teamId(df)
df = clean.convert_booleans(df)
df = clean.convert_lastPlayTime(df)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42,
                                                    shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.append(X_train, np.expand_dims(y_train, axis=1), axis=1)
X_test = np.append(X_test, np.expand_dims(y_test, axis=1), axis=1)

np.save('data/processed/train_static', X_train)
np.save('data/processed/test_static', X_test)

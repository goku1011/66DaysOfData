import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

spotify = pd.read_csv('spotify.csv')

X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))


# Base model to understand over-fitting and under-fitting
model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])
model.compile(
    optimizer='adam',
    loss='mae'
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
    verbose=0
)
history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss','val_loss']].plot()
plt.show()
# We see under-fitting of the model
print("Minimum validation loss {:0.4f}".format(history_df['val_loss'].min()))



# Adding some capacity to the base model to learn non-linearity
model = keras.Sequential([
    layers.Dense(units=128, activation='relu', input_shape=input_shape),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=1)
])
model.compile(
    optimizer='adam',
    loss='mae'
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
plt.show()
# We see over-fitting of the model
print("Minimum validation loss {:0.4f}".format(history_df['val_loss'].min()))



# Define Early Stopping to avoid over-fitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,    # minimium amount of change to count as an improvement
    patience=5,         # how many epochs to wait before stopping
    restore_best_weights=True,
)
model = keras.Sequential([
    layers.Dense(units=128, activation='relu', input_shape=input_shape),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=1)
])
model.compile(
    optimizer='adam',
    loss='mae'
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
    callbacks=[early_stopping],
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
plt.show()
print("Minimum validation loss {:0.4f}".format(history_df['val_loss'].min()))

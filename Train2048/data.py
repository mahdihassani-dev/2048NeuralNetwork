import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import random

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else: 
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype(np.float64)
            end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Set random seed for numpy
np.random.seed(42)

# Set random seed for Python's built-in random module
random.seed(42)

# Set random seed for TensorFlow
tf.random.set_seed(42)

# Load your dataset
data = pd.read_csv("VeryHugeDatasetReach4096.csv")
data = reduce_mem_usage(data)

# Separate inputs (X) and outputs (y)
X = data.iloc[:, 0:16]  # Input features
y = pd.get_dummies(data.iloc[:, -1])  # Convert output to one-hot encoding

# Define the categories for one-hot encoding (labels from 0 to 10)
categories = [i for i in range(12)]

# Apply one-hot encoding to each input column
X_encoded = pd.concat(
    [
        pd.get_dummies(
            pd.Categorical(X[col], categories=categories), prefix=col, prefix_sep="_"
        )
        for col in X
    ],
    axis=1,
)

# Define the model
model = Sequential()
model.add(
    Dense(
        16 * 12,
        input_dim=16 * 12,
        activation="relu",
        kernel_initializer="glorot_uniform",
    )
)  # Bias included by default
model.add(BatchNormalization())  # Add batch normalization layer
model.add(
    Dense(256, activation="relu", kernel_initializer="glorot_uniform")
)  # Bias included by default
model.add(BatchNormalization())  # Add batch normalization layer
model.add(
    Dense(256, activation="relu", kernel_initializer="glorot_uniform")
)  # Bias included by default
model.add(BatchNormalization())  # Add batch normalization layer
model.add(
    Dense(256, activation="relu", kernel_initializer="glorot_uniform")
)  # Bias included by default
model.add(BatchNormalization())  # Add batch normalization layer
model.add(
    Dense(256, activation="relu", kernel_initializer="glorot_uniform")
)  # Bias included by default
model.add(BatchNormalization())  # Add batch normalization layer
model.add(Dense(4, activation="softmax"))  # Output layer with 4 nodes for directions


# Define early stopping callback to stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Compile the model with the Adam optimizer and default learning rate
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model with mini-batch gradient descent and additional callbacks
model.fit(
    X_encoded,
    y,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping],
)

model.save("4096_model.h5")

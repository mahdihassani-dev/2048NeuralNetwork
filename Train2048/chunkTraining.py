import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import random

# Set random seed for numpy
np.random.seed(42)

# Set random seed for Python's built-in random module
random.seed(42)

# Set random seed for TensorFlow
tf.random.set_seed(42)

# Load your dataset in chunks
chunk_size = 10000000  # Specify the chunk size according to your memory constraints
data_chunks = pd.read_csv("VeryHugeDatasetReach4096.csv", chunksize=chunk_size)

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

# Iterate over each chunk
for i, data_chunk in enumerate(data_chunks):
    print(f"Processing Chunk {i+1}")
    
    # Separate inputs (X) and outputs (y)
    X = data_chunk.iloc[:, 0:16]  # Input features
    y = pd.get_dummies(data_chunk.iloc[:, -1])  # Convert output to one-hot encoding

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

    # Train the model with the current chunk
    model.fit(
        X_encoded, y,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose = 2
    )

# Save the model
model.save("4096_model_combined.h5")

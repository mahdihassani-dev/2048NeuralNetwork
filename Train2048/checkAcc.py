import pandas as pd
from keras.models import load_model

# Load your dataset
data = pd.read_csv('mainDataSet.csv')

# Separate inputs (X) and outputs (y)
X = data.iloc[:, :-1]  # Input features
y_true = data.iloc[:, -1]  # True output labels

# Define the categories for one-hot encoding (labels from 0 to 10)
categories = [i for i in range(11)]

# Apply one-hot encoding to each input column
X_encoded = pd.concat([pd.get_dummies(pd.Categorical(X[col], categories=categories), prefix=col, prefix_sep='_') for col in X], axis=1)

# Load the trained model
model = load_model('2048_model_one_hot.h5')


# Prepare the input data (apply the same preprocessing steps as during training)
# For example, you may need to scale or normalize the input features
# Make sure to preprocess the input data in the same way as during training

# Make predictions on the input data
y_pred_prob = model.predict(X_encoded)
y_pred = y_pred_prob.argmax(axis=1)

# Evaluate the predictions
accuracy = (y_pred == y_true).mean()
print("Accuracy:", accuracy)

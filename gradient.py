import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import requests
import joblib

# Load data from ThingSpeak account
def load_data_from_thingspeak(channel_id, api_key):
    url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}'
    response = requests.get(url)
    data = response.json()

    # Convert data to a DataFrame
    df = pd.DataFrame(data['feeds'])

    return df

# Function to extract relevant features from the data
def extract_features(data):
    features = data[['field1', 'field2', 'field3', 'field4', 'field5']].values.tolist()
    return features

# Handle missing values
def handle_missing_values(X):
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)
    return X

# Load the trained models
def load_models():
    models = []
    for i in range(5):
        model = joblib.load(f'gradient_boosting_regressor_model_{i+1}.pkl')
        models.append(model)
    return models

# Load the models
models = load_models()

# ThingSpeak configuration
channel_id = '2190048'
api_key = 'TYKCZAR5YS1YQY74'

# Fetch data from ThingSpeak
data = load_data_from_thingspeak(channel_id, api_key)

# Extract features from the data
features = extract_features(data)

# Convert features to numpy array
X = np.array(features)

# Handle missing values in the features
X = handle_missing_values(X)

# Make predictions for each target variable
predicted_values = np.array([model.predict(X) for model in models])

# Print the predicted values for each field
field_names = ['Temperature', 'Humidity', 'Pressure', 'Light Intensity', 'Altitude']
for field, values in zip(field_names, predicted_values):
    print(f"Predicted {field}: {values}")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Load the trained model
def load_model():
    model = joblib.load('random_forest_regressor_model.pkl')
    return model

# Load the model
model = load_model()

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

# Make predictions
predicted_values = model.predict(X)

# Print the predicted values for each field
field_names = ['Temperature', 'Humidity', 'Pressure', 'Light Intensity', 'Altitude']
for field, values in zip(field_names, predicted_values.T):
    print(f"Predicted {field}: {values}")

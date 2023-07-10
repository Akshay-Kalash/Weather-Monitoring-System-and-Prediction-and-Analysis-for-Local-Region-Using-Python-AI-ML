import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

# Load data from CSV file
data = pd.read_csv("weather2.csv")

# Function to extract relevant features from the data
def extract_features(data):
    features = data[['temperature', 'humidity', 'pressure', 'light_intensity', 'altitude']].values.tolist()
    return features

# Function to extract labels from the data
def extract_labels(data):
    labels = data[['temperature', 'humidity', 'pressure', 'light_intensity', 'altitude']].values.tolist()
    return labels

# Handle missing values
def handle_missing_values(X):
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)
    return X

# Extract features and labels from the data
features = extract_features(data)
labels = extract_labels(data)

# Convert features and labels to numpy arrays
X = np.array(features)
y = np.array(labels)

# Remove rows with NaN or infinite values
valid_indices = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
X = X[valid_indices]
y = y[valid_indices]

# Handle missing values in features
X = handle_missing_values(X)

# Train the AI model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a sample data for prediction
data = {
    'temperature': 25.6,
    'humidity': 65.2,
    'pressure': 1013.2,
    'light_intensity': 1200.0,
    'altitude': 100.5
}

# Convert the sample data to numpy array
X_pred = np.array([[data['temperature'], data['humidity'], data['pressure'], data['light_intensity'], data['altitude']]])

# Handle missing values in the prediction data
X_pred = handle_missing_values(X_pred)

# Make predictions
predicted_values = model.predict(X_pred)

# Print the predicted values for each field
field_names = ['Temperature', 'Humidity', 'Pressure', 'Light Intensity', 'Altitude']
for field, value in zip(field_names, predicted_values[0]):
    print(f"Predicted {field}: {value}")

# Save the trained model using joblib
joblib.dump(model, 'random_forest_regressor_model.pkl', compress=3)

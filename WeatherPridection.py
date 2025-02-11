import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset (Replace 'weather_data.csv' with your actual file)
data = pd.read_csv('weather_data.csv')

# Display first few rows to understand the data structure
print(data.head())

# Handle missing values if any
data = data.dropna()

# Selecting features and target variable
# Assume 'Temperature' is the target, and other columns like 'Humidity', 'WindSpeed', 'Pressure' are features
features = ['Humidity', 'WindSpeed', 'Pressure']  # Replace with actual feature names
target = 'Temperature'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Save the model for future use
joblib.dump(best_model, 'weather_prediction_model.pkl')

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Function to predict weather for new data
def predict_weather(humidity, wind_speed, pressure):
    input_data = np.array([[humidity, wind_speed, pressure]])
    predicted_temp = best_model.predict(input_data)
    return predicted_temp[0]

# Example prediction
humidity = float(input("Enter humidity (%): "))
wind_speed = float(input("Enter wind speed (km/h): "))
pressure = float(input("Enter pressure (hPa): "))

predicted_temperature = predict_weather(humidity, wind_speed, pressure)
print(f'Predicted Temperature: {predicted_temperature:.2f} °C')
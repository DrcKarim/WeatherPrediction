import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset (Replace 'weather_data.csv' with your actual file)
data = pd.read_csv('weather_data.csv')

# Display first few rows to understand the data structure
print(data.head())

# Selecting features and target variable
# Assume 'Temperature' is the target, and other columns like 'Humidity', 'WindSpeed' are features
features = ['Humidity', 'WindSpeed', 'Pressure']  # Replace with actual feature names
target = 'Temperature'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to predict weather for new data
def predict_weather(humidity, wind_speed, pressure):
    input_data = np.array([[humidity, wind_speed, pressure]])
    predicted_temp = model.predict(input_data)
    return predicted_temp[0]

# Example prediction
humidity = float(input("Enter humidity (%): "))
wind_speed = float(input("Enter wind speed (km/h): "))
pressure = float(input("Enter pressure (hPa): "))

predicted_temperature = predict_weather(humidity, wind_speed, pressure)
print(f'Predicted Temperature: {predicted_temperature:.2f} °C')

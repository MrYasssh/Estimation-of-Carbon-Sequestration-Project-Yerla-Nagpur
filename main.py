import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("carbon_sequestration_extended_dataset.csv")

# Drop non-numeric columns that are not features
df = df.drop(columns=["Timestamp"])  # Removing Timestamp since it's not a numerical feature

# Encode categorical variables
df = pd.get_dummies(df, columns=["Species", "Weather Condition"], drop_first=True)

# Feature selection
features = [col for col in df.columns if col not in ["Carbon Sequestration (kg CO₂/15 min)", "Oxygen Release (kg O₂/15 min)"]]
target = "Carbon Sequestration (kg CO₂/15 min)"

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

# Predictions
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")

# Function to predict sequestration and oxygen release
def predict_sequestration(species, weather, age, temperature, humidity, soil_nutrients, rainfall, cloudiness, watering, height, trunk_diameter, foliage_diameter):
    input_data = pd.DataFrame(columns=features)
    input_data.loc[0] = 0  # Initialize with zeros
    input_data["Tree Age (years)"] = age
    input_data["Temperature (°C)"] = temperature
    input_data["Humidity (%)"] = humidity
    input_data["Soil Nutrients (%)"] = soil_nutrients
    input_data["Rainfall (mm)"] = rainfall
    input_data["Cloudiness Index"] = cloudiness
    input_data["Watering Frequency (times/week)"] = watering
    input_data["Tree Height (m)"] = height
    input_data["Trunk Diameter (m)"] = trunk_diameter
    input_data["Foliage Diameter (m)"] = foliage_diameter
    
    species_col = f"Species_{species}"
    weather_col = f"Weather Condition_{weather}"
    
    if species_col in input_data.columns:
        input_data[species_col] = 1
    if weather_col in input_data.columns:
        input_data[weather_col] = 1
    
    input_data = input_data[features]  # Ensure correct feature ordering
    
    sequestration = model.predict(input_data)[0]
    oxygen_release = sequestration * 0.7  # Approximate ratio
    
    return sequestration, oxygen_release

# Example usage
species = "Neem"
weather = "Rainy"
age = 30
temperature = 28
humidity = 70
soil_nutrients = 6
rainfall = 20
cloudiness = 3
watering = 3
height = 15
trunk_diameter = 0.8
foliage_diameter = 6

carbon_sequestration, oxygen_release = predict_sequestration(species, weather, age, temperature, humidity, soil_nutrients, rainfall, cloudiness, watering, height, trunk_diameter, foliage_diameter)
print(f"Predicted Carbon Sequestration: {carbon_sequestration:.2f} kg CO₂/15 min")
print(f"Estimated Oxygen Release: {oxygen_release:.2f} kg O₂/15 min")

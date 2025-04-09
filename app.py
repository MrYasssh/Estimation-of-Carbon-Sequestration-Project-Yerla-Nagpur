import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Carbon Sequestration Predictor", layout="wide")

# Load dataset
df = pd.read_csv("carbon_sequestration_extended_dataset.csv")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center;'>🌱 Carbon Sequestration: A Case Study of Heartfulness Camp, Yerla</h1>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("🌿 **Input Parameters**")
species = st.sidebar.selectbox("🌳 **Select Tree Species**", df["Species"].unique())
weather = st.sidebar.selectbox("⛅ **Weather Condition**", df["Weather Condition"].unique())
age = st.sidebar.slider("🌱 **Tree Age (years)**", int(df["Tree Age (years)"].min()), int(df["Tree Age (years)"].max()), int(df["Tree Age (years)"].median()))
temperature = st.sidebar.slider("🌡 **Temperature (°C)**", int(df["Temperature (°C)"].min()), int(df["Temperature (°C)"].max()), int(df["Temperature (°C)"].median()))
humidity = st.sidebar.slider("💧 **Humidity (%)**", int(df["Humidity (%)"].min()), int(df["Humidity (%)"].max()), int(df["Humidity (%)"].median()))
soil_nutrients = st.sidebar.slider("🌿 **Soil Nutrients (%)**", int(df["Soil Nutrients (%)"].min()), int(df["Soil Nutrients (%)"].max()), int(df["Soil Nutrients (%)"].median()))
rainfall = st.sidebar.slider("🌧 **Rainfall (mm)**", int(df["Rainfall (mm)"].min()), int(df["Rainfall (mm)"].max()), int(df["Rainfall (mm)"].median()))
cloudiness = st.sidebar.slider("☁ **Cloudiness Index**", int(df["Cloudiness Index"].min()), int(df["Cloudiness Index"].max()), int(df["Cloudiness Index"].median()))
watering = st.sidebar.slider("💦 **Watering Frequency (times/week)**", int(df["Watering Frequency (times/week)"].min()), int(df["Watering Frequency (times/week)"].max()), int(df["Watering Frequency (times/week)"].median()))
height = st.sidebar.slider("🌳 **Tree Height (m)**", int(df["Tree Height (m)"].min()), int(df["Tree Height (m)"].max()), int(df["Tree Height (m)"].median()))
trunk_diameter = st.sidebar.slider("🪵 **Trunk Diameter (m)**", float(df["Trunk Diameter (m)"].min()), float(df["Trunk Diameter (m)"].max()), float(df["Trunk Diameter (m)"].median()))
foliage_diameter = st.sidebar.slider("🌿 **Foliage Diameter (m)**", float(df["Foliage Diameter (m)"].min()), float(df["Foliage Diameter (m)"].max()), float(df["Foliage Diameter (m)"].median()))

# Prediction function
def predict_sequestration():
    input_data = pd.DataFrame([{ 
        "Tree Age (years)": age, "Temperature (°C)": temperature, "Humidity (%)": humidity, "Soil Nutrients (%)": soil_nutrients, 
        "Rainfall (mm)": rainfall, "Cloudiness Index": cloudiness, "Watering Frequency (times/week)": watering, "Tree Height (m)": height, 
        "Trunk Diameter (m)": trunk_diameter, "Foliage Diameter (m)": foliage_diameter
    }])
    
    # Encode categorical variables (Species & Weather)
    for col in df.columns:
        if col.startswith("Species_") or col.startswith("Weather Condition_"):
            input_data[col] = 0  # Initialize all species/weather condition columns to 0
    
    species_col = f"Species_{species}"
    weather_col = f"Weather Condition_{weather}"
    
    if species_col in input_data.columns:
        input_data[species_col] = 1
    if weather_col in input_data.columns:
        input_data[weather_col] = 1
    
    # Ensure the order of features matches the trained model
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    
    sequestration = model.predict(input_data)[0]
    oxygen_release = sequestration * 0.7  # Approximate ratio
    
    return sequestration, oxygen_release

# Predict button
if st.sidebar.button("🌍 Predict Carbon Sequestration"):
    sequestration, oxygen = predict_sequestration()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🌿 **Predicted Carbon Sequestration:** {sequestration:.2f} kg CO₂/15 min")
    with col2:
        st.info(f"💨 **Estimated Oxygen Release:** {oxygen:.2f} kg O₂/15 min")

    # Display results graphically
    fig = px.bar(df, x="Species", y="Carbon Sequestration (kg CO₂/15 min)",
                 title="🌳 Carbon Sequestration Across Different Species", labels={"y": "Carbon Sequestration (kg CO₂/15 min)", "x": "Species"},
                 color="Species")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 Relationship Between Tree Parameters")
    fig2 = px.scatter_3d(df, x="Tree Age (years)", y="Tree Height (m)", z="Carbon Sequestration (kg CO₂/15 min)",
                          color="Species", size="Foliage Diameter (m)",
                          title="3D Visualization of Tree Parameters and Carbon Sequestration")
    st.plotly_chart(fig2, use_container_width=True)

# Footer Section
st.markdown("""
    <hr>
    <p style='text-align: center;'>🪴 An interdisciplinary project utilizing machine learning to predict carbon sequestration of campus trees.</p>
    <p style='text-align: center;'>Developed by students of the Department of Artificial Intelligence & Data Science, KDKCE.</p>
""", unsafe_allow_html=True)

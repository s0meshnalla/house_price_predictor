import streamlit as st
import pickle
import pandas as pd

# Load the model
MODEL_PATH = "LR_Model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file not found. Please upload 'LR_Model.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load location data
DATA_PATH = "cleaned_data.csv"
try:
    df = pd.read_csv(DATA_PATH)
    locations = sorted(df["location"].unique().tolist())  # Get unique locations
except FileNotFoundError:
    st.error("Error: Dataset file not found. Please upload 'cleaned_data.csv'.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.title("🏡 House Price Prediction")

# Get input features from user
st.sidebar.header("Enter House Details")

total_sqft = st.sidebar.number_input("Total Square Feet", min_value=100, max_value=10000, value=1000)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
BHK = st.sidebar.number_input("Number of BHK", min_value=1, max_value=10, value=3)
location = st.sidebar.selectbox("Select Location", locations)  # Dropdown for locations

# Ensure correct format
input_data = pd.DataFrame([[total_sqft, bath, BHK, location]],
                           columns=["total_sqft", "bath", "BHK", "location"])

# Predict
if st.sidebar.button("Predict Price"):
    try:
        prediction = model.predict(input_data)  # Ensure input is a DataFrame
        st.write("### 🏠 Predicted House Price: ₹", round(prediction[0], 2)*100000)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

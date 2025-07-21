import joblib
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("nigeria_house_price_model.pkl")

# Define the list of states (remove duplicates and sort)
states = sorted(list(set([
    "Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano",
    'Abia', 'Akwa Ibom', 'Anambar', 'Anambara',
    'Bayelsa', 'Borno', 'Cross River', 'Delta',
    'Edo','Ekiti','Enugu','Imo','Kaduna', 'Katsina','Kogi',
    'Kwara','Nasarawa','Niger','Ogun','Osun','Oyo','Plateau','Rivers'
])))

# Initialize and fit LabelEncoder
le = LabelEncoder()
le.fit(states)

# --- Streamlit App ---

st.title("🏠 Nigeria House Price Prediction App")
st.markdown("### Predict the price of a house in Nigeria based on its features")
st.image("https://example.com/house_image.jpg", use_container_width=True)
st.sidebar.title("🏠 Nigeria House Price Prediction App")

st.divider()

st.write("This app predicts the price of a house in Nigeria based on several features, " \
"such as bedrooms, bathrooms, toilets, parking spaces, and the state where the house is located.")

st.divider()

# --- Input Features ---
st.subheader("🏗️ Enter House Features")

bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=9)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=9)
toilets = st.number_input("Number of Toilets", min_value=1, max_value=9)
parking_space = st.number_input("Number of Parking Spaces", min_value=1, max_value=9)

state = st.selectbox("State", options=le.classes_)
state_encoded = le.transform([state])[0]

# --- Make prediction ---
st.divider()
if st.button("🔍 Predict House Price!"):
    # Prepare input for model
    X = np.array([[bedrooms, bathrooms, toilets, parking_space, state_encoded]])
    
    # Predict
    prediction = model.predict(X)[0]*1000000  # Convert to Naira

    # Display result
    st.balloons()
    st.success(f"💰 Predicted House Price: ₦{prediction:,.2f}")
else:
    st.info("Click the 'Predict House Price!' button to estimate the house price.")

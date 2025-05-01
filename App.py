import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import time

st.set_page_config(page_title="Delivery Time Prediction App", page_icon="📦")

# Load trained model
with open("models/delivery_time_n_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_columns = model_data["features"]

# App Header Image
img = Image.open("./assets/delivery_app.png")
img_resized = img.resize((380, 150))
st.image(img_resized)

# Title and Description
st.title("📦 Delivery Time Prediction App ")
st.caption("This is a Streamlit-based web application that predicts delivery time for an order based on multiple input features like shipping method, customer location, weather, distance, and more. The prediction model is trained using Python's scikit-learn library and serialized using pickle.")

# Sidebar Input Fields
with st.sidebar:
    st.header("👉 Input Parameters")

    product_category = st.selectbox("🛍️ ***Product Category***", ["Electronics", "Clothing", "Home & Kitchen", "Books", "Other"])
    customer_location = st.selectbox("📍 ***Customer Location***", ["Urban", "Suburban", "Rural"])
    shipping_method = st.selectbox("🚚 ***Shipping Method***", ["Standard", "Express", "Same-Day"])
    shipping_priority = st.selectbox("⚡ ***Shipping Priority***", ["Normal", "High", "Urgent"])
    weather = st.selectbox("🌤️ ***Weather Condition***", ["Sunny", "Rainy", "Snowy", "Stormy"])
    weight = st.number_input("⚖️ ***Package Weight (kg)***", min_value=0.1, max_value=100.0, step=0.1)
    package_size = st.selectbox("📏 ***Package Size***", ["Small", "Medium", "Large"])
    distance = st.number_input("📍 ***Distance to Destination (km)***", min_value=1, max_value=5000, step=1)
    warehouse_available = st.radio("🏢 ***Available in Nearby Warehouse?***", ["Yes", "No"])
    delivery_type = st.selectbox("🏠 ***Delivery Type***", ["Residential", "Commercial"])

    submit = st.button(label="**🚀 Predict Delivery Time**")

# Output Section
with st.container():
    if submit:
        st.subheader("📊 Predicted Delivery Time")

        input_dict = {
            "weight": [weight],
            "distance": [distance],
            "product_category": [product_category],
            "customer_location": [customer_location],
            "shipping_method": [shipping_method],
            "shipping_priority": [shipping_priority],
            "weather": [weather],
            "package_size": [package_size],
            "warehouse_available": [warehouse_available],
            "delivery_type": [delivery_type]
        }

        input_df = pd.DataFrame(input_dict)

        # Encode input to match training format
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        
        # Prediction
        prediction = model.predict(input_encoded)[0]
        prediction = round(prediction)

        with st.spinner("Predicting..."):
            time.sleep(1)
            st.success(f"📦 ***Estimated Delivery Time***:  **{prediction} days**")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by [Darshan Pardeshi](https://github.com/darshan1654/Delivery-Time-Prediction-App)")

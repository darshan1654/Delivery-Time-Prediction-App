# 📦 Delivery Time Prediction App

This is a **Streamlit-based web application** that predicts delivery time for an order based on multiple input features like shipping method, customer location, weather, distance, and more. The prediction model is trained using Python's `scikit-learn` library and serialized using `pickle`.

---

## 🚀 Features

- Predicts delivery time in days.
- Uses a trained machine learning model stored in a `.pkl` file.
- Clean and user-friendly UI built with Streamlit.
- Input features include: `Product Category`, `Customer Location`, `Shipping Method`, `Shipping Priority`, `Weather Condition`, `Package Weight`, `Package Size`, `Distance`, `Warehouse Availability`, `Delivery Type`
- Output feature: `Delivery Time`

## 🧠 Model Training

The model is trained using the following pipeline:

- Data preprocessing using `pandas` and `sklearn`
- Feature scaling using `StandardScaler`
- Model: `LinearRegression`
- Evaluation metrics: `MAE`, `R²`
- Trained model is saved as `delivery_time_model.pkl`

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/darshan1654/Delivery-Time-Prediction-App.git
cd Delivery-Time-Prediction-App
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run App.py
```

## **📝** License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

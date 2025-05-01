# 📦 Delivery Time Prediction App

This is a **Streamlit-based web application** that predicts delivery time for an order based on multiple input features like shipping method, customer location, weather, distance, and more. The prediction model is trained using Python's `scikit-learn` library and serialized using `pickle`.

---

## 🚀 Features

- Predicts delivery time in days.
- Uses a trained machine learning model stored in a `.pkl` file.
- Clean and user-friendly UI built with Streamlit.
- Input features include:
  - Product Category
  - Customer Location
  - Shipping Method
  - Shipping Priority
  - Weather Condition
  - Package Weight
  - Package Size
  - Distance to Destination
  - Nearby Warehouse Availability
  - Delivery Type

---

## 🧠 Model Training

The model is trained using the following pipeline:

- Data preprocessing using `pandas` and `sklearn`
- Feature scaling using `StandardScaler`
- Model: `LinearRegression`
- Evaluation metrics: `MAE`, `R²`
- Trained model is saved as `wait_time_model2.pkl`

---

## 📁 File Structure

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/your-username/delivery-time-prediction-app.git
cd delivery-time-prediction-app
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run App.py
```

---

## **📝** License
MIT License.

---

## **🙌** Acknowledgements
Streamlit for frontend development

Scikit-learn for model building

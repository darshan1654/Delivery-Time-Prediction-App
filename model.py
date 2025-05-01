#delivery_training_model.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Simulate dataset
n = 5000
np.random.seed(42)

data = pd.DataFrame({
    "product_category": np.random.choice(["Electronics", "Clothing", "Home & Kitchen", "Books", "Other"], n),
    "customer_location": np.random.choice(["Urban", "Suburban", "Rural"], n),
    "shipping_method": np.random.choice(["Standard", "Express", "Same-Day"], n),
    "shipping_priority": np.random.choice(["Normal", "High", "Urgent"], n),
    "weather": np.random.choice(["Sunny", "Rainy", "Snowy", "Stormy"], n),
    "weight": np.round(np.random.uniform(0.1, 50, n), 2),
    "package_size": np.random.choice(["Small", "Medium", "Large"], n),
    "distance": np.random.randint(10, 3000, n),
    "warehouse_available": np.random.choice(["Yes", "No"], n),
    "delivery_type": np.random.choice(["Residential", "Commercial"], n)
})

# Apply logic to simulate target variable (delivery_time)
def simulate_delivery_time(row):
    # Base time
    if row.shipping_method == "Standard":
        time = np.random.randint(3, 7)
    elif row.shipping_method == "Express":
        time = np.random.randint(2, 4)
    else:
        time = 1

    # Adjustments
    if row.customer_location == "Rural":
        time += 2
    if row.weather in ["Rainy", "Snowy", "Stormy"]:
        time += 1
    if row.weight > 10:
        time += 1
    if row.distance > 1000:
        time += 2
    if row.package_size == "Large":
        time += 1
    if row.warehouse_available == "Yes":
        time -= 1
    if row.delivery_type == "Commercial":
        time -= 1
    if row.shipping_priority == "Urgent":
        time = min(time, 2)

    return max(1, time)

data["delivery_time"] = data.apply(simulate_delivery_time, axis=1)

# Encode categorical variables
data_enc = pd.get_dummies(data.drop(columns=["delivery_time"]), drop_first=True)
y = data["delivery_time"]

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(data_enc, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
with open("models/delivery_time_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": data_enc.columns.tolist()}, f)

print("✅ Model trained successfully and saved!!")

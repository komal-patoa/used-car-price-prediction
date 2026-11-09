
import re
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
DATA_PATH = "data/train-data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# 3. Clean numeric columns containing units
def extract_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+\.?\d*", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan

df["Mileage"] = df["Mileage"].apply(extract_number)
df["Engine"] = df["Engine"].apply(extract_number)
df["Power"] = df["Power"].apply(extract_number)
df["New_Price"] = df["New_Price"].apply(extract_number)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Kilometers_Driven"] = pd.to_numeric(
    df["Kilometers_Driven"], errors="coerce"
)

# 4. Extract car brand from car name
df["Brand"] = df["Name"].fillna("Unknown").str.split().str[0]

# 5. Remove original Name column
df = df.drop(columns=["Name"])

# 6. Separate input features and target
X = df.drop(columns=["Price"])
y = pd.to_numeric(df["Price"], errors="coerce")

# Remove rows where target price is missing or invalid
valid_rows = y.notna() & (y >= 0)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 7. Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)

# 8. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 9. Create ML model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ))
])

# 10. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining model...")

# 11. Train
model.fit(X_train, y_train)

# 12. Evaluate
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", round(mae, 3), "Lakh")
print("RMSE:", round(rmse, 3), "Lakh")
print("R2 Score:", round(r2, 3))

# 13. Save model and feature information
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": model,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}, "models/car_price_model.pkl")

print("\nModel saved successfully!")
print("File: models/car_price_model.pkl")
joblib.dump(model, "models/car_price_model.pkl")
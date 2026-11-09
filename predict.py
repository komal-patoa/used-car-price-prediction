
import joblib
import pandas as pd

from features import prepare_features


# Load trained model
model = joblib.load("models/car_price_model.pkl")

# Load test dataset
test_data = pd.read_csv("data/test-data.csv")

# Keep original data for final output
original_data = test_data.copy()

# Prepare test features
X_test = prepare_features(test_data)

# Predict car prices
predictions = model.predict(X_test)

# Add predictions to output
original_data["Predicted_Price"] = predictions

# Save predictions
original_data.to_csv(
    "predictions.csv",
    index=False
)

print("Predictions completed successfully!")

print("\nFirst 10 predictions:")
print(
    original_data[
        ["Name", "Predicted_Price"]
    ].head(10)
)

print("\nSaved file: predictions.csv")

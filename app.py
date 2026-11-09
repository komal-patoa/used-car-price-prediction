
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from features import prepare_features


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "train-data.csv"


# ==================================================
# LOAD MODEL AND DATASET
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🚗 Car Price Prediction")

st.sidebar.write("Machine Learning Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Price",
        "Model Performance",
        "About Project"
    ]
)


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.title("🚗 Used Car Price Prediction Dashboard")

    st.write(
        "Explore the used car dataset and understand "
        "different factors affecting car prices."
    )

    st.divider()

    # Price statistics
    total_cars = len(df)

    average_price = df["Price"].mean()

    minimum_price = df["Price"].min()

    maximum_price = df["Price"].max()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cars",
        f"{total_cars:,}"
    )

    col2.metric(
        "Average Price",
        f"{average_price:.2f}"
    )

    col3.metric(
        "Minimum Price",
        f"{minimum_price:.2f}"
    )

    col4.metric(
        "Maximum Price",
        f"{maximum_price:.2f}"
    )

    st.divider()

    # Price distribution
    st.subheader("📊 Price Distribution")

    price_bins = pd.cut(
        df["Price"],
        bins=20
    )

    price_distribution = (
        price_bins.value_counts()
        .sort_index()
    )

    chart_data = pd.DataFrame({
        "Price Range": price_distribution.index.astype(str),
        "Number of Cars": price_distribution.values
    })

    st.bar_chart(
        chart_data,
        x="Price Range",
        y="Number of Cars"
    )

    st.divider()

    # Fuel type analysis
    st.subheader("⛽ Cars by Fuel Type")

    fuel_data = df["Fuel_Type"].value_counts()

    st.bar_chart(fuel_data)

    st.divider()

    # Transmission analysis
    st.subheader("⚙️ Cars by Transmission")

    transmission_data = df["Transmission"].value_counts()

    st.bar_chart(transmission_data)

    st.divider()

    # Average price by fuel
    st.subheader("💰 Average Price by Fuel Type")

    fuel_price = (
        df.groupby("Fuel_Type")["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(fuel_price)

    st.divider()

    # Top brands
    st.subheader("🏆 Top 10 Car Brands")

    df["Brand"] = (
        df["Name"]
        .fillna("Unknown")
        .astype(str)
        .str.split()
        .str[0]
    )

    brand_data = df["Brand"].value_counts().head(10)

    st.bar_chart(brand_data)

    st.divider()

    # Dataset preview
    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ==================================================
# PREDICT PRICE PAGE
# ==================================================

elif page == "Predict Price":

    st.title("🔮 Predict Used Car Price")

    st.write(
        "Enter the car details below to estimate its price."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        car_name = st.text_input(
            "Car Name",
            "Maruti Swift VDI"
        )

        location = st.selectbox(
            "Location",
            sorted(df["Location"].dropna().unique())
        )

        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=2020
        )

        kilometers = st.number_input(
            "Kilometers Driven",
            min_value=0,
            value=30000
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(df["Fuel_Type"].dropna().unique())
        )

        transmission = st.selectbox(
            "Transmission",
            sorted(df["Transmission"].dropna().unique())
        )

    with col2:

        owner_type = st.selectbox(
            "Owner Type",
            sorted(df["Owner_Type"].dropna().unique())
        )

        mileage = st.number_input(
            "Mileage",
            min_value=0.0,
            value=20.0
        )

        engine = st.number_input(
            "Engine (CC)",
            min_value=0,
            value=1200
        )

        power = st.number_input(
            "Power (BHP)",
            min_value=0.0,
            value=80.0
        )

        seats = st.selectbox(
            "Seats",
            [2, 4, 5, 6, 7, 8, 9, 10],
            index=2
        )

        new_price = st.number_input(
            "New Price",
            min_value=0.0,
            value=8.0
        )

    st.divider()

    if st.button(
        "🚗 Predict Car Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{

            "Name": car_name,
            "Location": location,
            "Year": year,
            "Kilometers_Driven": kilometers,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Type": owner_type,
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats,
            "New_Price": new_price

        }])

        try:

            processed_data = prepare_features(input_data)

            prediction = model.predict(processed_data)[0]

            st.success("Prediction Completed Successfully!")

            st.subheader("Estimated Car Price")

            st.metric(
                "Predicted Price",
                f"{prediction:.2f}"
            )

            st.info(
                "This is an estimated price generated "
                "by the Machine Learning model."
            )

        except Exception as error:

            st.error(
                f"Prediction Error: {error}"
            )


# ==================================================
# MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.title("📈 Model Performance & Evaluation")

    st.write(
        "Evaluate how well the Random Forest "
        "Regression model predicts used car prices."
    )

    st.divider()

    # Prepare features and target
    X = df.drop(columns=["Price"])

    y = df["Price"]

    X = prepare_features(X)

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model predictions
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Display metrics
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    st.divider()

    # Actual vs predicted graph
    st.subheader("🎯 Actual Price vs Predicted Price")

    comparison = pd.DataFrame({

        "Actual Price": y_test.values,

        "Predicted Price": predictions

    })

    st.scatter_chart(
        comparison,
        x="Actual Price",
        y="Predicted Price"
    )

    st.divider()

    # Prediction comparison
    st.subheader("📋 Actual vs Predicted Values")

    st.dataframe(
        comparison.head(20).round(2),
        use_container_width=True
    )

    # Download results
    csv_data = comparison.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Evaluation Results",
        data=csv_data,
        file_name="model_evaluation.csv",
        mime="text/csv"
    )

    st.caption(
        "Evaluation uses 20% held-out data from the training dataset."
    )


# ==================================================
# ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.write("""
    ### Used Car Price Prediction

    This project predicts the estimated selling price
    of a used car using Machine Learning.

    ### Machine Learning Algorithm

    Random Forest Regression

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Joblib

    ### Project Workflow

    1. Dataset Collection
    2. Data Cleaning
    3. Missing Value Handling
    4. Feature Engineering
    5. Model Training
    6. Model Evaluation
    7. Price Prediction
    8. Streamlit Dashboard

    ### Project Features

    - Dataset Analysis
    - Price Statistics
    - Fuel Type Analysis
    - Transmission Analysis
    - Used Car Price Prediction
    - Model Performance Evaluation
    - Downloadable Evaluation Results
    """)

    st.divider()

    st.write("**Project:** Used Car Price Prediction")

    st.write("**Algorithm:** Random Forest Regressor")

    st.write("**Interface:** Streamlit Web Application")

    st.write("**Category:** Machine Learning")

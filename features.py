
import re
import pandas as pd


def extract_number(value):
    """Extract the first numeric value from text."""
    if pd.isna(value):
        return float("nan")

    match = re.search(r"[\d,]+\.?\d*", str(value))

    if match:
        return float(match.group().replace(",", ""))

    return float("nan")


def prepare_features(data):
    """Clean and prepare used-car dataset features."""

    df = data.copy()

    # Remove unnecessary index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Extract brand from car name
    if "Name" in df.columns:
        df["Brand"] = (
            df["Name"]
            .fillna("Unknown")
            .astype(str)
            .str.split()
            .str[0]
        )

    # Convert text-based numerical columns
    numeric_text_columns = [
        "Mileage",
        "Engine",
        "Power",
        "New_Price"
    ]

    for column in numeric_text_columns:
        if column in df.columns:
            df[column] = df[column].apply(extract_number)

    # Convert other numerical columns
    numeric_columns = [
        "Year",
        "Kilometers_Driven",
        "Seats"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    return df

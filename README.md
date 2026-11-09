
# 🚗 Used Car Price Prediction

### Machine Learning Based Car Price Analysis and Prediction Dashboard

## 📌 Project Overview

Used Car Price Prediction is a Machine Learning project that predicts the estimated selling price of second-hand cars.

The system uses different car details such as brand, manufacturing year, kilometers driven, fuel type, transmission, mileage, engine capacity and power to estimate the car price.

This project uses the **Random Forest Regression** algorithm and provides an interactive web application developed using Streamlit.

---

## 🎯 Project Objectives

- Analyse used car dataset.
- Clean and preprocess the data.
- Identify important factors affecting car prices.
- Train a Machine Learning model.
- Predict the estimated selling price of used cars.
- Evaluate model performance.
- Develop an interactive web dashboard.

---

## 📊 Dataset Information

The project uses a used car dataset containing approximately **6,019 training records**.

### Important Features

| Feature | Description |
|---|---|
| Name | Car name |
| Location | Car location |
| Year | Manufacturing year |
| Kilometers_Driven | Distance travelled |
| Fuel_Type | Type of fuel |
| Transmission | Manual or Automatic |
| Owner_Type | Ownership details |
| Mileage | Car mileage |
| Engine | Engine capacity |
| Power | Engine power |
| Seats | Number of seats |
| New_Price | Original new car price |
| Price | Selling price (Target Variable) |

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- Streamlit
- Joblib
- Visual Studio Code

---

## 🤖 Machine Learning Algorithm

### Random Forest Regression

Random Forest Regression is an ensemble Machine Learning algorithm that combines multiple Decision Trees to predict numerical values.

In this project:

1. Multiple Decision Trees are created.
2. Each tree learns patterns from the training dataset.
3. Each tree predicts a car price.
4. The predictions are combined to generate the final estimated price.

### Why Random Forest?

- Handles complex data relationships.
- Works with numerical and categorical features.
- Can provide good prediction performance.
- Reduces overfitting compared with a single decision tree in many cases.

---

## 🔄 Project Workflow

1. Dataset Collection
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Data Preprocessing
6. Train-Test Split (80:20)
7. Random Forest Model Training
8. Model Evaluation
9. Model Saving using Joblib
10. Streamlit Web Application
11. Used Car Price Prediction

---

## 📈 Model Performance

The model was evaluated using 20% held-out testing data.

| Evaluation Metric | Result |
|---|---:|
| MAE | 1.54 |
| RMSE | 3.86 |
| R² Score | 0.8790 |

### Evaluation Metrics Explanation

- **MAE:** Measures the average absolute difference between actual and predicted prices.
- **RMSE:** Measures prediction errors and gives more importance to larger errors.
- **R² Score:** Indicates how much variation in car prices is explained by the model.

The model achieved an R² score of **0.8790**, which indicates that it explains approximately 87.9% of the variation in the target variable on the held-out test data.

---

## ✨ Application Features

The Streamlit application contains four main sections:

### 1. Dashboard
- Total number of cars.
- Average, minimum and maximum prices.
- Car price distribution.
- Fuel type analysis.
- Transmission analysis.
- Average price by fuel type.
- Top 10 car brands.
- Dataset preview.

### 2. Predict Price
- Enter car details.
- Select fuel type and transmission.
- Enter manufacturing year and kilometers driven.
- Get estimated car price.

### 3. Model Performance
- MAE evaluation.
- RMSE evaluation.
- R² score.
- Actual vs Predicted Price graph.
- Download evaluation results.

### 4. About Project
- Project description.
- Technologies used.
- Machine Learning workflow.
- Project information.

---

## 📁 Project Structure

```text
car prize prediction/
│
├── data/
│   ├── train-data.csv
│   └── test-data.csv
│
├── models/
│   └── car_price_model.pkl
│
├── features.py
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open Project Folder

```bash
cd "car prize prediction"
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Train the Model

```bash
python train.py
```

This creates the trained model file inside the models folder.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 📦 Requirements

The project uses the following Python libraries:

```text
numpy
pandas
scikit-learn
streamlit
joblib
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

- Add more recent used car data.
- Include car condition and accident history.
- Compare multiple Machine Learning algorithms.
- Improve model performance through hyperparameter tuning.
- Deploy the application online.
- Add more useful visualizations.

---

## 🎓 Project Conclusion

This project demonstrates the practical application of Machine Learning in solving a real-world problem.

The Random Forest Regression model analyses different car features and predicts estimated selling prices.

The Streamlit dashboard makes the system interactive and easy to use.

---

## 👩‍💻 Project Information

**Project Name:** Used Car Price Prediction

**Domain:** Machine Learning

**Algorithm:** Random Forest Regression

**Programming Language:** Python

**Interface:** Streamlit

**IDE:** Visual Studio Code

---

### ⭐ If you find this project useful, feel free to star the repository!

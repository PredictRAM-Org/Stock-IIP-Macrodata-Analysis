import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Function to merge economic data with company-specific data
def merge_data(company_data, economic_data, macro_data):
    merged_data = pd.merge(company_data, economic_data, on='Date', how='left')
    merged_data = pd.merge(merged_data, macro_data, on='Reporting Date', how='left')
    return merged_data

# Function to train and evaluate regression models
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regression': RandomForestRegressor(),
        'Gradient Boosting Regression': GradientBoostingRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results[name] = {'model': model, 'mse': mse}

    return results

# Function to make predictions
def make_predictions(model, input_data):
    return model.predict([input_data])

# Streamlit app
def main():
    st.title("Revenue Prediction Dashboard")

    # Upload files
    st.sidebar.header("Upload Files")
    company_file = st.sidebar.file_uploader("Upload Company Data (xlsx)", type=["xlsx"])
    iip_file = st.sidebar.file_uploader("Upload IIP Data (csv)", type=["csv"])
    macro_file = st.sidebar.file_uploader("Upload Macro Data (xlsx)", type=["xlsx"])

    # Load data
    if company_file is not None and iip_file is not None and macro_file is not None:
        company_data = load_data(company_file)
        iip_data = load_data(iip_file)
        macro_data = load_data(macro_file)

        # Merge data
        merged_data = merge_data(company_data, iip_data, macro_data)

        # Select columns for analysis
        selected_columns = st.multiselect("Select Columns for Analysis", merged_data.columns)

        # Train-test split
        X = merged_data[selected_columns].dropna()
        y = merged_data['Total Revenue/Income'].loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        results = train_models(X_train, y_train, X_test, y_test)

        # Display model results
        st.subheader("Model Results")
        for name, result in results.items():
            st.write(f"{name} MSE: {result['mse']}")

        # User input for prediction
        st.sidebar.header("Make Predictions")
        input_data = {}
        for column in selected_columns:
            input_data[column] = st.sidebar.number_input(f"Enter value for {column}", min_value=0)

        # User selects model for prediction
        selected_model = st.sidebar.selectbox("Select Model for Prediction", list(results.keys()))
        selected_model = results[selected_model]['model']

        # Make predictions
        if st.sidebar.button("Make Predictions"):
            prediction = make_predictions(selected_model, input_data)
            st.subheader("Predicted Total Revenue/Income")
            st.write(f"The predicted value is: {prediction[0]}")

if __name__ == "__main__":
    main()

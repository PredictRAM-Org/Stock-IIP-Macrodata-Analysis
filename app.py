import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load data
def load_data(file_path):
    if file_path is not None:
        return pd.read_excel(file_path)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if file_path is None


# Merge data
def merge_data(company_data, iip_data, macro_data):
    merged_data = pd.merge(company_data, iip_data, on='Date', how='left')
    merged_data = pd.merge(merged_data, macro_data, on='Reporting Date', how='left')
    return merged_data

# Train models
def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regression': RandomForestRegressor(),
        'Gradient Boosting Regression': GradientBoostingRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = model

    return results

# Make predictions
def make_predictions(model, input_data):
    return model.predict([input_data])

# Streamlit app
def main():
    st.title("Revenue Prediction Dashboard")

    # Upload files
    company_file = st.file_uploader("Upload Company Data (xlsx)", type=["xlsx"])
    iip_file = st.file_uploader("Upload IIP Data (csv)", type=["csv"])
    macro_file = st.file_uploader("Upload Macro Data (xlsx)", type=["xlsx"])

    if company_file is not None and iip_file is not None and macro_file is not None:
        # Load data
        company_data = load_data(company_file)
        iip_data = load_data(iip_file)
        macro_data = load_data(macro_file)

        # Merge data
        merged_data = merge_data(company_data, iip_data, macro_data)

        # Select columns for analysis
        selected_columns = st.multiselect("Select Columns for Analysis", merged_data.columns)

        # Train-test split
        X = merged_data[selected_columns].dropna()
        y = X.pop('Total Revenue/Income')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        models = train_models(X_train, y_train)

        # User input for prediction
        st.sidebar.header("Make Predictions")
        input_data = {}
        for column in selected_columns:
            input_data[column] = st.sidebar.number_input(f"Enter value for {column}", min_value=0)

        # User selects model for prediction
        selected_model = st.sidebar.selectbox("Select Model for Prediction", list(models.keys()))

        # Make predictions
        if st.sidebar.button("Make Predictions"):
            prediction = make_predictions(models[selected_model], input_data)
            st.subheader("Predicted Total Revenue/Income")
            st.write(f"The predicted value is: {prediction[0]}")

if __name__ == "__main__":
    main()

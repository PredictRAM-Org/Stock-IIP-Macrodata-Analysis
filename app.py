# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from io import StringIO

# Step 1: Read macro_data.xlsx file
macro_data = pd.read_excel('macro_data.xlsx')

# Step 2: Read IIP.csv file
iip_data = pd.read_csv('IIP.csv')

# Step 3: Create a Streamlit app
st.title("Economic Indicators Predictive Model")

# Step 4: Add widgets for user input
selected_macro_columns = st.multiselect('Select columns from macro_data.xlsx:', macro_data.columns)
selected_iip_columns = st.multiselect('Select columns from IIP.csv:', iip_data.columns)

# Step 5: Upload stock_data file
uploaded_stock_data = st.file_uploader("Upload stock_data Excel file", type=["xlsx", "xls"])

# Check if a file was uploaded
if uploaded_stock_data is not None:
    # Read uploaded stock_data file
    stock_data = pd.read_excel(uploaded_stock_data)

    # Check if the 'Total Revenue/Income' column exists in stock_data
    if 'Total Revenue/Income' in stock_data.columns:
        # Step 6: Train predictive models (Linear Regression, Random Forest Regression, Gradient Boosting Regression)
        # Example for macro_data
        macro_selected_data = macro_data[selected_macro_columns]

        # Convert the index to datetime (assuming the index is the date column)
        macro_selected_data.index = pd.to_datetime(macro_selected_data.index)

        # Aggregate monthly data to quarterly
        macro_selected_data_quarterly = macro_selected_data.resample('Q').mean()

        macro_target = stock_data['Total Revenue/Income']

        # Ensure the lengths of input arrays are consistent
        if len(macro_selected_data_quarterly) == len(macro_target):
            # Split data into training and testing sets
            macro_train_data, macro_test_data, macro_train_target, macro_test_target = train_test_split(
                macro_selected_data_quarterly, macro_target, test_size=0.2, random_state=42
            )

            # Linear Regression
            linear_model = LinearRegression()
            linear_model.fit(macro_train_data, macro_train_target)
            linear_predictions = linear_model.predict(macro_test_data)

            # Random Forest Regression
            rf_model = RandomForestRegressor()
            rf_model.fit(macro_train_data, macro_train_target)
            rf_predictions = rf_model.predict(macro_test_data)

            # Gradient Boosting Regression
            gb_model = GradientBoostingRegressor()
            gb_model.fit(macro_train_data, macro_train_target)
            gb_predictions = gb_model.predict(macro_test_data)

            # Step 7: Compare models with income statement data
            # Compare models using Mean Squared Error
            linear_mse = mean_squared_error(macro_test_target, linear_predictions)
            rf_mse = mean_squared_error(macro_test_target, rf_predictions)
            gb_mse = mean_squared_error(macro_test_target, gb_predictions)

            st.write(f'Linear Regression Mean Squared Error: {linear_mse}')
            st.write(f'Random Forest Regression Mean Squared Error: {rf_mse}')
            st.write(f'Gradient Boosting Regression Mean Squared Error: {gb_mse}')

            # Step 8: Add input options for upcoming values
            upcoming_iip_data = st.text_input('Enter upcoming values for selected IIP.csv columns:')
            upcoming_macro_data = st.text_input('Enter upcoming values for selected macro_data.xlsx columns:')

            # Step 9: Predict upcoming total revenue/income, total operating expense, EBITDA, and Net income
            # Example for predicting upcoming values
            upcoming_iip_data = pd.read_csv(StringIO(upcoming_iip_data))
            upcoming_macro_data = pd.read_excel(StringIO(upcoming_macro_data))

            # Use trained models to predict upcoming values
            upcoming_macro_predictions = linear_model.predict(upcoming_macro_data)

            # Step 10: Display comparative chart
            # Code for displaying chart goes here

            # Optionally, you can display the predicted values
            st.write("Predicted Total Revenue/Income:", upcoming_macro_predictions)
        else:
            st.error("Lengths of input arrays are inconsistent. Please check your data.")
    else:
        st.error("Selected column 'Total Revenue/Income' not found in the uploaded stock_data file.")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

# Load stock income statement data
try:
    stock_data = pd.read_json('stock_income_data.json')
except Exception as e:
    st.error(f"Error loading stock data: {e}")

# Load IIP data
try:
    iip_data = pd.read_csv('IIP.csv')
except Exception as e:
    st.error(f"Error loading IIP data: {e}")

# Load macro data
try:
    macro_data = pd.read_excel('macro_data.xlsx')
except Exception as e:
    st.error(f"Error loading macro data: {e}")

# Merge stock data with IIP data on date
try:
    merged_data = pd.merge(stock_data, iip_data, on='Date', how='inner')
except Exception as e:
    st.error(f"Error merging stock and IIP data: {e}")

# Merge merged_data with macro data on Reporting Date
try:
    final_data = pd.merge(merged_data, macro_data, left_on='Date', right_on='Reporting Date', how='inner')
except Exception as e:
    st.error(f"Error merging final data: {e}")

# Feature engineering - add more features based on relationships between economic indicators and income statements
# ...

# User input for upcoming expected values
user_input_iip = st.number_input("Enter upcoming IIP value", value=0.0)
user_input_macro = st.number_input("Enter upcoming macro data value", value=0.0)

# Predictive modeling
features = final_data.drop(['Total Revenue', 'Total Operating Expense', 'EBITDA', 'Net Income'], axis=1)
target_columns = ['Total Revenue', 'Total Operating Expense', 'EBITDA', 'Net Income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, final_data[target_columns], test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_prediction = lr_model.predict([[user_input_iip, user_input_macro]])
st.write("Linear Regression Prediction:", lr_prediction)

# Random Forest Regression
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict([[user_input_iip, user_input_macro]])
st.write("Random Forest Regression Prediction:", rf_prediction)

# Gradient Boosting Regression
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_prediction = gb_model.predict([[user_input_iip, user_input_macro]])
st.write("Gradient Boosting Regression Prediction:", gb_prediction)

# Display comparative chart
try:
    fig, ax = plt.subplots()
    ax.plot(final_data['Date'], final_data['Total Revenue'], label='Actual Total Revenue')
    ax.plot(final_data['Date'], lr_model.predict(features), label='Linear Regression Prediction')
    ax.plot(final_data['Date'], rf_model.predict(features), label='Random Forest Regression Prediction')
    ax.plot(final_data['Date'], gb_model.predict(features), label='Gradient Boosting Regression Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error displaying chart: {e}")

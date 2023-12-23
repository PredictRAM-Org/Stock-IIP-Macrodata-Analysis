import streamlit as st
import pandas as pd

# Function to load all JSON files from a folder
def load_json_files_from_folder(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    dataframes = [pd.read_json(os.path.join(folder_path, file)) for file in json_files]
    return pd.concat(dataframes, ignore_index=True)

# Function to upload and read stock data from Excel file
def upload_and_read_stock_data():
    uploaded_file = st.file_uploader("Upload stock data Excel file", type=["xlsx"])
    if uploaded_file is not None:
        stock_data = pd.read_excel(uploaded_file)
        return stock_data
    else:
        st.warning("Please upload a stock data Excel file.")
        return pd.DataFrame()

# Load stock income statement data from the 'stock_data' folder
try:
    stock_data_folder_path = 'stock_data'
    stock_data = load_json_files_from_folder(stock_data_folder_path)
except Exception as e:
    st.error(f"Error loading stock data: {e}")
    stock_data = pd.DataFrame()

# Load IIP data
try:
    iip_data = pd.read_csv('IIP.csv')
except Exception as e:
    st.error(f"Error loading IIP data: {e}")
    iip_data = pd.DataFrame()

# Load macro data
try:
    macro_data = pd.read_excel('macro_data.xlsx')
except Exception as e:
    st.error(f"Error loading macro data: {e}")
    macro_data = pd.DataFrame()

# Display information about available data
st.write("Available Stock Data:")
st.write(stock_data.head())

st.write("Available IIP Data:")
st.write(iip_data.head())

st.write("Available Macro Data:")
st.write(macro_data.head())

# Display stock data based on user upload
stock_data_upload = upload_and_read_stock_data()

# Check if data loading was successful before proceeding
if not (stock_data.empty or iip_data.empty or macro_data.empty or stock_data_upload.empty):
    # Merge stock data with IIP data on date
    try:
        merged_data = pd.merge(stock_data_upload, iip_data, on='Date', how='inner')
    except Exception as e:
        st.error(f"Error merging stock and IIP data: {e}")
        merged_data = pd.DataFrame()

    # Merge merged_data with macro data on Reporting Date
    try:
        final_data = pd.merge(merged_data, macro_data, left_on='Date', right_on='Reporting Date', how='inner')
    except Exception as e:
        st.error(f"Error merging final data: {e}")
        final_data = pd.DataFrame()

    # Check if merging was successful before proceeding
    if not (merged_data.empty or final_data.empty):
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
else:
    st.error("Data loading was unsuccessful. Please check your data files.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load Data
df = pd.read_excel('C:\\Users\\00078411\\OneDrive - GMR Holdings Private Limited\\Desktop\\Coal_Bidder_Data.xlsx')  # Replace with your actual data

# Clean the 'Month' and 'Year' columns
df['Month'] = df['Month'].apply(lambda x: str(x).strip())
df['Year'] = df['Year'].astype(str).apply(lambda x: x.split('.')[0])  # Handle cases where Year might be a float

# Ensure 'Month' is in a format that can be parsed
df['Month'] = df['Month'].apply(lambda x: x.capitalize())

# Create a 'Date' column by combining 'Month' and 'Year'
df['Date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'], errors='coerce')

# Handle any rows where the 'Date' couldn't be parsed
df = df.dropna(subset=['Date'])

# Extract additional date-related features
df['Month_Num'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# Ensure 'Year' is treated as a numerical feature
df['Year'] = df['Year'].astype(int)

# Handle missing values
df['Grade'] = df['Grade'].fillna('Unknown')
df['Quantity Offered'] = df['Quantity Offered'].fillna(df['Quantity Offered'].mean())
df['Final Allocation'] = df['Final Allocation'].fillna(df['Final Allocation'].mean())  # This line imputes missing values

# Convert categorical variables to numeric using Label Encoding
le_mine = LabelEncoder()
df['Mine Name'] = le_mine.fit_transform(df['Mine Name'])

le_grade = LabelEncoder()
df['Grade'] = le_grade.fit_transform(df['Grade'])

# Ensure there are no missing values in the target variables
df = df.dropna(subset=['Final Allocation', 'Price'])

# Features for Price Prediction (excluding Final Allocation and Quarter)
X_price = df[['Mine Name', 'Grade', 'Allocated Qty', 'Quantity Offered', 'Month_Num', 'Year']]
y_price = df['Price']

# Features for Final Allocation Prediction (excluding Allocated Qty and Quarter)
X_allocation = df[['Mine Name', 'Quantity Offered', 'Month_Num', 'Year']]
y_allocation = df['Final Allocation']

# Split the data for Price Prediction
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

# Model Training for Price Prediction
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_price_train, y_price_train)

# Predict on test data for Price Prediction
y_price_pred = price_model.predict(X_price_test)

# Evaluate the Price Prediction model
price_mae = mean_absolute_error(y_price_test, y_price_pred)

# Split the data for Final Allocation Prediction
X_allocation_train, X_allocation_test, y_allocation_train, y_allocation_test = train_test_split(X_allocation, y_allocation, test_size=0.2, random_state=42)

# Model Training for Final Allocation Prediction
allocation_model = RandomForestRegressor(n_estimators=100, random_state=42)
allocation_model.fit(X_allocation_train, y_allocation_train)

# Predict on test data for Final Allocation Prediction
y_allocation_pred = allocation_model.predict(X_allocation_test)

# Evaluate the Final Allocation Prediction model
allocation_mae = mean_absolute_error(y_allocation_test, y_allocation_pred)

# Streamlit App
st.title("Coal Market Predictions")

# Input fields for prediction
mine_name_input = st.selectbox("Mine Name", le_mine.classes_)
grade_input = st.selectbox("Grade", le_grade.classes_)
allocated_qty_input = st.number_input("Allocated Quantity", min_value=0)
quantity_offered_input = st.number_input("Quantity Offered", min_value=0)
month_input = st.selectbox("Month", range(1, 13))
year_input = st.number_input("Year", min_value=2000, max_value=2100, step=1)

# Convert inputs to appropriate numeric values using encoders
mine_name_enc = le_mine.transform([mine_name_input])[0]
grade_enc = le_grade.transform([grade_input])[0]

# Prediction for Coal Price
if st.button("Predict Coal Price"):
    input_data_price = np.array([[mine_name_enc, grade_enc, allocated_qty_input, quantity_offered_input, month_input, year_input]])
    predicted_price = price_model.predict(input_data_price)
    st.write(f"Predicted Coal Price: {predicted_price[0]}")
    st.write(f"Model Mean Absolute Error: {price_mae}")

# Prediction for Final Allocation
if st.button("Predict Final Allocation"):
    input_data_allocation = np.array([[mine_name_enc, quantity_offered_input, month_input, year_input]])
    predicted_allocation = allocation_model.predict(input_data_allocation)
    st.write(f"Predicted Final Allocation: {predicted_allocation[0]}")
    st.write(f"Model Mean Absolute Error: {allocation_mae}")

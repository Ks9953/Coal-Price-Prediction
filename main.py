import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Cache the data loading function
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

# Cache the data preprocessing function
@st.cache_data
def preprocess_data(df):
    # Clean the 'Month' and 'Year' columns
    df['Month'] = df['Month'].apply(lambda x: str(x).strip().capitalize())
    df['Year'] = df['Year'].astype(str).apply(lambda x: x.split('.')[0])
    df['Date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Month_Num'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Year'].astype(int)
    df['Grade'] = df['Grade'].fillna('Unknown')
    df['Quantity Offered'] = df['Quantity Offered'].fillna(df['Quantity Offered'].mean())
    df['Final Allocation'] = df['Final Allocation'].fillna(df['Final Allocation'].mean())
    le_mine = LabelEncoder()
    df['Mine Name'] = le_mine.fit_transform(df['Mine Name'])
    le_grade = LabelEncoder()
    df['Grade'] = le_grade.fit_transform(df['Grade'])
    df = df.dropna(subset=['Final Allocation', 'Price'])
    return df, le_mine, le_grade

# Cache the model training function
@st.cache_resource
def train_models(df):
    # Split the data for Price Prediction
    X_price = df[['Mine Name', 'Grade', 'Allocated Qty', 'Quantity Offered', 'Month_Num', 'Year']]
    y_price = df['Price']
    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
    price_model = RandomForestRegressor(n_estimators=100, random_state=42)
    price_model.fit(X_price_train, y_price_train)
    y_price_pred = price_model.predict(X_price_test)
    price_mae = mean_absolute_error(y_price_test, y_price_pred)
    
    # Split the data for Final Allocation Prediction
    X_allocation = df[['Mine Name', 'Quantity Offered', 'Month_Num', 'Year']]
    y_allocation = df['Final Allocation']
    X_allocation_train, X_allocation_test, y_allocation_train, y_allocation_test = train_test_split(X_allocation, y_allocation, test_size=0.2, random_state=42)
    allocation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    allocation_model.fit(X_allocation_train, y_allocation_train)
    y_allocation_pred = allocation_model.predict(X_allocation_test)
    allocation_mae = mean_absolute_error(y_allocation_test, y_allocation_pred)
    
    return price_model, price_mae, allocation_model, allocation_mae

# Streamlit app
def main():
    st.title("Coal Market Predictions")

    # Load and preprocess data
    df = load_data('Coal_Data.xlsx')
    df, le_mine, le_grade = preprocess_data(df)
    price_model, price_mae, allocation_model, allocation_mae = train_models(df)

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

if __name__ == "__main__":
    main()

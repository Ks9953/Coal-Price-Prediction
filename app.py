from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load Data
df = pd.read_excel('C:\\Users\\00078411\\OneDrive - GMR Holdings Private Limited\\Desktop\\Coal_Bidder_Data.xlsx')  # Replace with your actual data

# Data Preprocessing
df['Month'] = df['Month'].apply(lambda x: str(x).strip())
df['Year'] = df['Year'].astype(str).apply(lambda x: x.split('.')[0])  # Handle cases where Year might be a float
df['Month'] = df['Month'].apply(lambda x: x.capitalize())
df['Date'] = pd.to_datetime(df['Month'] + ' ' + df['Year'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Month_Num'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Year'].astype(int)
df['Grade'] = df['Grade'].fillna('Unknown')
df['Quantity Offered'] = df['Quantity Offered'].fillna(df['Quantity Offered'].mean())
df['Final Allocation'] = df['Final Allocation'].fillna(df['Final Allocation'].mean())

# Convert categorical variables to numeric using Label Encoding
le_mine = LabelEncoder()
df['Mine Name'] = le_mine.fit_transform(df['Mine Name'])

le_grade = LabelEncoder()
df['Grade'] = le_grade.fit_transform(df['Grade'])

df = df.dropna(subset=['Final Allocation', 'Price'])

# Features and target variable
X_price = df[['Mine Name', 'Grade', 'Allocated Qty', 'Quantity Offered', 'Month_Num', 'Year']]
y_price = df['Price']

X_allocation = df[['Mine Name', 'Quantity Offered', 'Month_Num', 'Year']]
y_allocation = df['Final Allocation']

# Train/Test Split and Model Training
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_price_train, y_price_train)
y_price_pred = price_model.predict(X_price_test)
price_mae = mean_absolute_error(y_price_test, y_price_pred)

X_allocation_train, X_allocation_test, y_allocation_train, y_allocation_test = train_test_split(X_allocation, y_allocation, test_size=0.2, random_state=42)
allocation_model = RandomForestRegressor(n_estimators=100, random_state=42)
allocation_model.fit(X_allocation_train, y_allocation_train)
y_allocation_pred = allocation_model.predict(X_allocation_test)
allocation_mae = mean_absolute_error(y_allocation_test, y_allocation_pred)

@app.route('/')
def index():
    mine_names = le_mine.classes_
    grades = le_grade.classes_
    return render_template('index.html', mine_names=mine_names, grades=grades)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    mine_name_input = request.form['mine_name']
    grade_input = request.form['grade']
    allocated_qty_input = int(request.form['allocated_qty'])
    quantity_offered_input = int(request.form['quantity_offered'])
    month_input = int(request.form['month'])
    year_input = int(request.form['year'])

    mine_name_enc = le_mine.transform([mine_name_input])[0]
    grade_enc = le_grade.transform([grade_input])[0]

    input_data_price = np.array([[mine_name_enc, grade_enc, allocated_qty_input, quantity_offered_input, month_input, year_input]])
    predicted_price = price_model.predict(input_data_price)[0]

    return render_template('result.html', result_type="Price", predicted_value=predicted_price, mae=price_mae)

@app.route('/predict_allocation', methods=['POST'])
def predict_allocation():
    mine_name_input = request.form['mine_name']
    quantity_offered_input = int(request.form['quantity_offered'])
    month_input = int(request.form['month'])
    year_input = int(request.form['year'])

    mine_name_enc = le_mine.transform([mine_name_input])[0]

    input_data_allocation = np.array([[mine_name_enc, quantity_offered_input, month_input, year_input]])
    predicted_allocation = allocation_model.predict(input_data_allocation)[0]

    return render_template('result.html', result_type="Final Allocation", predicted_value=predicted_allocation, mae=allocation_mae)

if __name__ == '__main__':
    app.run(debug=True)

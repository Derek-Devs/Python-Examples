import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
# Assuming we have a CSV file with the sales data
data = pd.read_csv('sales_data.csv', parse_dates=['date'])

# Feature engineering
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day_of_week'] = data['date'].dt.dayofweek

# Aggregate sales data
agg_data = data.groupby(['year', 'month', 'day_of_week', 'product_id', 'warehouse_id']).agg({'quantity': 'sum'}).reset_index()

# Step 2: Feature engineering
X = agg_data[['year', 'month', 'day_of_week', 'product_id', 'warehouse_id']]
y = agg_data['quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train a predictive model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Plotting actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Quantity')
plt.title('Actual vs Predicted Sales Quantity')
plt.show()

# Step 5: Make predictions for the next month (example)
# Assuming we want to predict for the next month (e.g., January 2024)
future_data = pd.DataFrame({
    'year': [2024] * 7,
    'month': [1] * 7,
    'day_of_week': list(range(7)),
    'product_id': [1] * 7,
    'warehouse_id': [1] * 7
})

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)
print(f'Future Predictions: {future_predictions}')

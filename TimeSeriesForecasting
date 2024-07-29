import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Step 1: Load and preprocess the data
# Assuming we have a CSV file with the sales data
data = pd.read_csv('daily_sales.csv', parse_dates=['date'], index_col='date')

# Check for missing values and handle them
data.isnull().sum()
data.dropna(inplace=True)

# Step 2: Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(data['sales'], label='Daily Sales')
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 3: Decompose the time series to identify trends and seasonality
decomposition = seasonal_decompose(data['sales'], model='additive', period=365)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()

# Step 4: Train a time series forecasting model (ARIMA)
# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Fit the ARIMA model
model = ARIMA(train['sales'], order=(5, 1, 0)) # Adjust the order (p, d, q) as needed
model_fit = model.fit()

# Step 5: Evaluate the model
# Make predictions
predictions = model_fit.forecast(steps=len(test))
test['predictions'] = predictions

# Calculate evaluation metrics
mae = mean_absolute_error(test['sales'], test['predictions'])
mse = mean_squared_error(test['sales'], test['predictions'])
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Plot actual vs predicted sales
plt.figure(figsize=(12, 6))
plt.plot(train['sales'], label='Train Sales')
plt.plot(test['sales'], label='Test Sales')
plt.plot(test['predictions'], label='Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 6: Make future predictions
# Fit the model on the entire dataset and forecast future sales
final_model = ARIMA(data['sales'], order=(5, 1, 0))
final_model_fit = final_model.fit()
future_steps = 30  # Number of days to forecast
future_predictions = final_model_fit.forecast(steps=future_steps)

# Create a DataFrame for future dates
future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, closed='right')
future_df = pd.DataFrame({'date': future_dates, 'predicted_sales': future_predictions})

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(data['sales'], label='Historical Sales')
plt.plot(future_df['date'], future_df['predicted_sales'], label='Future Predicted Sales')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Save the future predictions
future_df.to_csv('future_sales_predictions.csv', index=False)

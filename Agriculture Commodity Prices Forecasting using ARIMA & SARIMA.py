# pip install pandas numpy matplotlib seaborn statsmodels scikit-learn torchvision openpyxl

################### Data Collection & Preprocessing #####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


# Load dataset
df = pd.read_excel('commodity_prices.xlsx')

print ("Rice Price Forecasting")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort and set index
df = df.sort_values(by=['Commodity', 'Date'])

# Remove duplicate index issue by resetting index
df = df.reset_index(drop=True)

# Fill missing values (Updated: Avoids deprecated `method` argument)
df.ffill(inplace=True)

# Group data by each commodity and ensure unique Date index for each group
commodity_groups = {}
for commodity in df['Commodity'].unique():
    subset = df[df['Commodity'] == commodity].copy()
    subset.set_index('Date', inplace=True)  # Set date as index for each commodity
    subset = subset.asfreq('D')  # Ensure daily frequency for time series
    subset.ffill(inplace=True)  # Forward-fill any missing values
    commodity_groups[commodity] = subset  # Store cleaned subset

df_rice = commodity_groups['Rice']

# Display dataset info
print(df_rice.info())
print(df_rice.head())
################### Exploratory Data Analysis (EDA) #####################

# Plot price trends for each commodity
plt.figure(figsize=(12, 6))
for commodity in df['Commodity'].unique():
    subset = df[df['Commodity'] == commodity]
    plt.plot(subset['Date'], subset['Price'], label=commodity)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Commodity Prices Over Time')
plt.legend()
plt.grid()
plt.show()

# Compute correlation only for numerical columns
numeric_cols = df.select_dtypes(include=[np.number])

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

######################## Forecasting Using ARIMA ########################

# Select a commodity (e.g., Rice) for ARIMA forecasting
df_rice = commodity_groups['Rice']

# Train-test split (80% train, 20% test)
train_size = int(len(df_rice) * 0.8)
train, test = df_rice.iloc[:train_size], df_rice.iloc[train_size:]

# Fit ARIMA model (Auto-tuned p,d,q)
arima_model = ARIMA(train['Price'], order=(3,1,3))  # Adjust based on ACF/PACF
arima_result = arima_model.fit()

# Forecast
forecast_arima = arima_result.predict(start=len(train), end=len(train) + len(test) - 1)
forecast_arima.index = test.index  # Align forecast index with test data

# Evaluate Model Performance
arima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_arima))
print(f'ARIMA RMSE: {arima_rmse:.4f}')

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Price', color='blue', marker='o')
plt.plot(test.index, forecast_arima, label='Predicted Price (ARIMA)', color='red', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Forecasting for Rice')
plt.legend()
plt.grid()
plt.show()

######################## Forecasting Using SARIMA ########################

# Fit SARIMA model (p,d,q) × (P,D,Q,s) where s=12 for monthly seasonality
sarima_model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
forecast_sarima = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)
forecast_sarima.index = test.index  # Align forecast index with test data

# Evaluate Model Performance
sarima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_sarima))
print(f'SARIMA RMSE: {sarima_rmse:.4f}')

# Plot actual vs predicted prices (SARIMA)
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Price', color='b', marker='o')
plt.plot(test.index, forecast_sarima, label='Predicted Price (SARIMA)', color='g', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SARIMA Forecasting for Rice')
plt.legend()
plt.grid()
plt.show()

################### Future Forecasting #####################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define future forecast duration (30 months)
future_months = 30

# Generate future dates starting from the last available month in the dataset
future_dates = pd.date_range(start=df_rice.index[-1], periods=future_months + 1, freq='M')[1:]

# Generate future predictions using trained ARIMA and SARIMA models
future_forecast_arima = arima_result.forecast(steps=future_months)
future_forecast_sarima = sarima_result.forecast(steps=future_months)

# Convert predictions into a DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'ARIMA Forecast (Rs)': future_forecast_arima.round(2),  # Rounded price to 2 decimal places
    'SARIMA Forecast (Rs)': future_forecast_sarima.round(2)
})

# Fix encoding issue in Windows terminal
try:
    print("Future Price Predictions for Rice (₹):".encode('utf-8').decode())
except UnicodeEncodeError:
    print("Future Price Predictions for Rice (Rs):")  # Fallback if ₹ is not supported

# Display first 900 future price predictions
print(future_df.head(900).to_string(index=False))

# Plot Future Forecast (30 Months)
plt.figure(figsize=(12, 6))
plt.plot(df_rice.index, df_rice['Price'], label='Historical Prices (Rs)', color='blue')
plt.plot(future_df['Date'], future_df['ARIMA Forecast (Rs)'], label='ARIMA Future Forecast (Rs)', linestyle='dashed', color='red')
plt.plot(future_df['Date'], future_df['SARIMA Forecast (Rs)'], label='SARIMA Future Forecast (Rs)', linestyle='dashed', color='green')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.title('Future Price Forecast for Rice (Next 30 Months)')
plt.legend()
plt.show()

################### Comparing ARIMA & SARIMA Forecasting #####################

print(f'ARIMA RMSE: {arima_rmse}')
print(f'SARIMA RMSE: {sarima_rmse}')

if sarima_rmse < arima_rmse:
    print("SARIMA performed better for price forecasting.")
else:
    print("ARIMA performed better for price forecasting.")

################### Data Collection & Preprocessing for Wheat #####################

# Load dataset
df = pd.read_excel('commodity_prices.xlsx')

print ("Wheat Price Forecasting")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort and set index
df = df.sort_values(by=['Commodity', 'Date'])

# Remove duplicate index issue by resetting index
df = df.reset_index(drop=True)

# Fill missing values
df.ffill(inplace=True)

# Group data by each commodity and ensure unique Date index for each group
commodity_groups = {}
for commodity in df['Commodity'].unique():
    subset = df[df['Commodity'] == commodity].copy()
    subset.set_index('Date', inplace=True)  # Set date as index for each commodity
    subset = subset.asfreq('D')  # Ensure daily frequency for time series
    subset.ffill(inplace=True)  # Forward-fill any missing values
    commodity_groups[commodity] = subset  # Store cleaned subset

# Select Wheat data
df_wheat = commodity_groups.get('Wheat', None)

if df_wheat is None:
    print("Error: No data available for 'Wheat' commodity!")
    exit()

# Display dataset info
print(df_wheat.info())
print(df_wheat.head())

######################## Forecasting Using ARIMA for Wheat ########################

# Train-test split (80% train, 20% test)
train_size = int(len(df_wheat) * 0.8)
train, test = df_wheat.iloc[:train_size], df_wheat.iloc[train_size:]

# Fit ARIMA model (Auto-tuned p,d,q)
arima_model = ARIMA(train['Price'], order=(3,1,3))  # Adjust based on ACF/PACF
arima_result = arima_model.fit()

# Forecast
forecast_arima = arima_result.predict(start=len(train), end=len(train) + len(test) - 1)
forecast_arima.index = test.index  # Align forecast index with test data

# Evaluate Model Performance
arima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_arima))
print(f'ARIMA RMSE: {arima_rmse:.4f}')

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Price', color='blue', marker='o')
plt.plot(test.index, forecast_arima, label='Predicted Price (ARIMA)', color='red', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Forecasting for Wheat')
plt.legend()
plt.grid()
plt.show()

######################## Forecasting Using SARIMA for Wheat ########################

# Fit SARIMA model (p,d,q) × (P,D,Q,s) where s=12 for monthly seasonality
sarima_model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
forecast_sarima = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)
forecast_sarima.index = test.index  # Align forecast index with test data

# Evaluate Model Performance
sarima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_sarima))
print(f'SARIMA RMSE: {sarima_rmse:.4f}')

# Plot actual vs predicted prices (SARIMA)
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Price', color='b', marker='o')
plt.plot(test.index, forecast_sarima, label='Predicted Price (SARIMA)', color='g', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SARIMA Forecasting for Wheat')
plt.legend()
plt.grid()
plt.show()

################### Future Forecasting for Wheat #####################

# Define future forecast duration (30 months)
future_months = 30

# Generate future dates starting from the last available month in the dataset
future_dates = pd.date_range(start=df_wheat.index[-1], periods=future_months + 1, freq='M')[1:]

# Generate future predictions using trained ARIMA and SARIMA models
future_forecast_arima = arima_result.forecast(steps=future_months)
future_forecast_sarima = sarima_result.forecast(steps=future_months)

# Convert predictions into a DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'ARIMA Forecast (Rs)': future_forecast_arima.round(2),
    'SARIMA Forecast (Rs)': future_forecast_sarima.round(2)
})

# Fix encoding issue in Windows terminal
try:
    print("Future Price Predictions for Wheat (₹):".encode('utf-8').decode())
except UnicodeEncodeError:
    print("Future Price Predictions for Wheat (Rs):")  # Fallback if ₹ is not supported

# Display first 900 future price predictions
print(future_df.head(900).to_string(index=False))

# Plot Future Forecast (30 Months)
plt.figure(figsize=(12, 6))
plt.plot(df_wheat.index, df_wheat['Price'], label='Historical Prices (Rs)', color='blue')
plt.plot(future_df['Date'], future_df['ARIMA Forecast (Rs)'], label='ARIMA Future Forecast (Rs)', linestyle='dashed', color='red')
plt.plot(future_df['Date'], future_df['SARIMA Forecast (Rs)'], label='SARIMA Future Forecast (Rs)', linestyle='dashed', color='green')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.title('Future Price Forecast for Wheat (Next 30 Months)')
plt.legend()
plt.show()

################### Comparing ARIMA & SARIMA Forecasting for Wheat #####################

print(f'ARIMA RMSE: {arima_rmse}')
print(f'SARIMA RMSE: {sarima_rmse}')

if sarima_rmse < arima_rmse:
    print("SARIMA performed better for price forecasting.")
else:
    print("ARIMA performed better for price forecasting.")


################### Data Collection & Preprocessing #####################

# Load dataset
df = pd.read_excel('commodity_prices.xlsx')

print ("Pulses Price Forecasting")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort and set index
df = df.sort_values(by=['Commodity', 'Date'])

# Remove duplicate index issue by resetting index
df = df.reset_index(drop=True)

# Fill missing values
df.ffill(inplace=True)

# Group data by each commodity and ensure unique Date index for each group
commodity_groups = {}
for commodity in df['Commodity'].unique():
    subset = df[df['Commodity'] == commodity].copy()
    subset.set_index('Date', inplace=True)  # Set date as index for each commodity
    subset = subset.asfreq('D')  # Ensure daily frequency for time series
    subset.ffill(inplace=True)  # Forward-fill any missing values
    commodity_groups[commodity] = subset  # Store cleaned subset

# Select Pulses data
df_pulses = commodity_groups.get('Pulses', None)

if df_pulses is None:
    print("Error: No data available for 'Pulses' commodity!")
    exit()

# Display dataset info
print(df_pulses.info())
print(df_pulses.head())

######################## Forecasting Using ARIMA for Pulses ########################

# Train-test split (80% train, 20% test)
train_size = int(len(df_pulses) * 0.8)
train, test = df_pulses.iloc[:train_size], df_pulses.iloc[train_size:]

# Fit ARIMA model (Auto-tuned p,d,q)
arima_model = ARIMA(train['Price'], order=(3,1,3))  # Adjust based on ACF/PACF
arima_result = arima_model.fit()

# Forecast
forecast_arima = arima_result.predict(start=len(train), end=len(train) + len(test) - 1)
forecast_arima.index = test.index  # Align forecast index with test data

# Evaluate Model Performance
arima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_arima))
print(f'ARIMA RMSE: {arima_rmse:.4f}')

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Price', color='blue', marker='o')
plt.plot(test.index, forecast_arima, label='Predicted Price (ARIMA)', color='red', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Forecasting for Pulses')
plt.legend()
plt.grid()
plt.show()

######################## Forecasting Using SARIMA for Pulses ########################

# Fit SARIMA model (p,d,q) × (P,D,Q,s) where s=12 for monthly seasonality
sarima_model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
forecast_sarima = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)
forecast_sarima.index = test.index  # Align forecast index with test data

# Evaluate Model Performance
sarima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_sarima))
print(f'SARIMA RMSE: {sarima_rmse:.4f}')

# Plot actual vs predicted prices (SARIMA)
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Price'], label='Actual Price', color='b', marker='o')
plt.plot(test.index, forecast_sarima, label='Predicted Price (SARIMA)', color='g', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SARIMA Forecasting for Pulses')
plt.legend()
plt.grid()
plt.show()

################### Future Forecasting for Pulses #####################

# Define future forecast duration (30 months)
future_months = 30

# Generate future dates starting from the last available month in the dataset
future_dates = pd.date_range(start=df_pulses.index[-1], periods=future_months + 1, freq='M')[1:]

# Generate future predictions using trained ARIMA and SARIMA models
future_forecast_arima = arima_result.forecast(steps=future_months)
future_forecast_sarima = sarima_result.forecast(steps=future_months)

# Convert predictions into a DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'ARIMA Forecast (Rs)': future_forecast_arima.round(2),
    'SARIMA Forecast (Rs)': future_forecast_sarima.round(2)
})

# Fix encoding issue in Windows terminal
try:
    print("Future Price Predictions for Pulses (₹):".encode('utf-8').decode())
except UnicodeEncodeError:
    print("Future Price Predictions for Pulses (Rs):")  # Fallback if ₹ is not supported

# Display first 900 future price predictions
print(future_df.head(900).to_string(index=False))

# Plot Future Forecast (30 Months)
plt.figure(figsize=(12, 6))
plt.plot(df_pulses.index, df_pulses['Price'], label='Historical Prices (Rs)', color='blue')
plt.plot(future_df['Date'], future_df['ARIMA Forecast (Rs)'], label='ARIMA Future Forecast (Rs)', linestyle='dashed', color='red')
plt.plot(future_df['Date'], future_df['SARIMA Forecast (Rs)'], label='SARIMA Future Forecast (Rs)', linestyle='dashed', color='green')
plt.xlabel('Date')
plt.ylabel('Price (Rs)')
plt.title('Future Price Forecast for Pulses (Next 30 Months)')
plt.legend()
plt.show()

################### Comparing ARIMA & SARIMA Forecasting for Pulses #####################

print(f'ARIMA RMSE: {arima_rmse}')
print(f'SARIMA RMSE: {sarima_rmse}')

if sarima_rmse < arima_rmse:
    print("SARIMA performed better for price forecasting.")
else:
    print("ARIMA performed better for price forecasting.")
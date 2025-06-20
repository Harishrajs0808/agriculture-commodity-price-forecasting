import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from django.http import FileResponse

# -------- Plotting Functions --------
def plot_commodity_trends(df):
    plt.figure(figsize=(12, 6))
    for commodity in df['Commodity'].unique():
        subset = df[df['Commodity'] == commodity]
        plt.plot(subset['Date'], subset['Price'], label=commodity)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Commodity Prices Over Time')
    plt.legend()
    plt.grid()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return chart_base64

def plot_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return chart_base64

def plot_chart(actual, pred_arima, pred_sarima, future_df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual['Price'], label='Actual Price', color='blue')
    plt.plot(pred_arima.index, pred_arima, label='ARIMA Forecast', linestyle='dashed', color='red')
    plt.plot(pred_sarima.index, pred_sarima, label='SARIMA Forecast', linestyle='dashed', color='green')
    plt.plot(future_df['Date'], future_df['ARIMA'], label='Future ARIMA', linestyle='dotted', color='red')
    plt.plot(future_df['Date'], future_df['SARIMA'], label='Future SARIMA', linestyle='dotted', color='green')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs)')
    plt.title(title)
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return chart_base64

# -------- Forecast Logic --------
def generate_forecast(df_commodity, commodity):
    train_size = int(len(df_commodity) * 0.8)
    train, test = df_commodity.iloc[:train_size], df_commodity.iloc[train_size:]

    try:
        arima_model = ARIMA(train['Price'], order=(3,1,3)).fit()
        forecast_arima = arima_model.predict(start=len(train), end=len(train) + len(test) - 1)
        forecast_arima.index = test.index
        arima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_arima))
    except Exception as e:
        print(f"ARIMA model failed for {commodity}: {e}")
        forecast_arima = pd.Series(index=test.index)
        arima_rmse = None

    try:
        sarima_model = SARIMAX(train['Price'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
        forecast_sarima = sarima_model.predict(start=len(train), end=len(train) + len(test) - 1)
        forecast_sarima.index = test.index
        sarima_rmse = np.sqrt(mean_squared_error(test['Price'], forecast_sarima))
    except Exception as e:
        print(f"SARIMA model failed for {commodity}: {e}")
        forecast_sarima = pd.Series(index=test.index)
        sarima_rmse = None

    last_date = df_commodity.index[-1].to_period('M').to_timestamp('M')
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=30, freq='M')

    future_forecast_arima = arima_model.forecast(steps=30) if arima_rmse else [None] * 30
    future_forecast_sarima = sarima_model.forecast(steps=30) if sarima_rmse else [None] * 30

    future_df = pd.DataFrame({
        'Date': future_dates,
        'ARIMA': np.round(future_forecast_arima, 2),
        'SARIMA': np.round(future_forecast_sarima, 2)
    })

    full_data = pd.concat([train, test])
    return full_data, forecast_arima, forecast_sarima, arima_rmse, sarima_rmse, future_df

# -------- Views --------
def forecast_view(request):
    forecasts = []
    commodity_trend_chart = None
    correlation_chart = None
    error = None

    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            df = pd.read_excel(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by=['Commodity', 'Date']).drop_duplicates()
            df.ffill(inplace=True)

            commodity_trend_chart = plot_commodity_trends(df)
            correlation_chart = plot_correlation_matrix(df)

            for commodity in df['Commodity'].unique():
                df_commodity = df[df['Commodity'] == commodity].set_index('Date').asfreq('M').ffill()
                full_data, pred_arima, pred_sarima, arima_rmse, sarima_rmse, future_df = generate_forecast(df_commodity, commodity)
                chart = plot_chart(full_data, pred_arima, pred_sarima, future_df, f"{commodity} Price Forecasting")
                forecasts.append({
                    'commodity': commodity,
                    'rmse_arima': round(arima_rmse, 2) if arima_rmse else 'N/A',
                    'rmse_sarima': round(sarima_rmse, 2) if sarima_rmse else 'N/A',
                    'chart': chart,
                    'future_df': future_df.to_html(classes="table table-bordered", index=False),
                })

        except Exception as e:
            error = f"Something went wrong: {e}"

    return render(request, 'forecast.html', {
        'forecasts': forecasts,
        'commodity_trend_chart': commodity_trend_chart,
        'correlation_chart': correlation_chart,
        'error': error
    })

def home_view(request):
    return render(request, 'index.html')

def dataset_view(request):
    return render(request, 'dataset.html')

def download_dataset(request):
    return render(request, 'download_dataset.html')


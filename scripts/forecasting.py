import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def prepare_time_series_data(df, target_column, lag_features, rolling_features):
    """
    Prepares time-series data with lag and rolling features.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The target column to predict.
        lag_features (list): List of lags to include (e.g., [1, 24]).
        rolling_features (list): List of rolling windows to calculate (e.g., [24, 48]).

    Returns:
        pd.DataFrame: Time-series data with additional features.
    """
    data = df[[target_column]].copy()
    for lag in lag_features:
        data[f'lag_{lag}'] = data[target_column].shift(lag)
    for window in rolling_features:
        data[f'rolling_mean_{window}'] = data[target_column].rolling(window=window).mean()
    data.dropna(inplace=True)
    return data


def train_forecasting_model(df, target_column, lag_features, rolling_features):
    """
    Trains a Random Forest model for time-series forecasting.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The target column to predict.
        lag_features (list): List of lags to include (e.g., [1, 24]).
        rolling_features (list): List of rolling windows to calculate (e.g., [24, 48]).

    Returns:
        RandomForestRegressor: The trained model.
        pd.DataFrame: The prepared time-series data.
    """
    data = prepare_time_series_data(df, target_column, lag_features, rolling_features)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    split_index = int(0.8 * len(data))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='orange')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, data


def forecast_future(model, last_known_data, lag_features, rolling_features, forecast_periods):
    """
    Forecasts future values using the trained model.

    Parameters:
        model (RandomForestRegressor): The trained model.
        last_known_data (pd.DataFrame): The last known data for feature generation.
        lag_features (list): List of lags to include (e.g., [1, 24]).
        rolling_features (list): List of rolling windows to calculate (e.g., [24, 48]).
        forecast_periods (int): Number of periods to forecast.

    Returns:
        pd.DataFrame: Forecasted values with timestamps.
    """
    forecast = []
    last_data = last_known_data.copy()

    for _ in range(forecast_periods):
        # Generate features for the next period
        new_row = {}
        for lag in lag_features:
            new_row[f'lag_{lag}'] = last_data[-lag:].mean()
        for window in rolling_features:
            new_row[f'rolling_mean_{window}'] = last_data[-window:].mean()

        # Convert to DataFrame for prediction
        new_row = pd.DataFrame([new_row])
        prediction = model.predict(new_row)[0]

        # Append prediction and update last_data
        forecast.append(prediction)
        last_data = np.append(last_data, prediction)[-max(lag_features):]

    forecast_dates = pd.date_range(start=last_known_data.index[-1], periods=forecast_periods, freq='H')
    forecast_df = pd.DataFrame({'Forecast Date': forecast_dates, 'Predicted Values': forecast})
    forecast_df.set_index('Forecast Date', inplace=True)

    return forecast_df
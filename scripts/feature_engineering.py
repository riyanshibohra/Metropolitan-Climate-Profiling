import pandas as pd

def calculate_seasonal_thresholds(df, columns, group_column='Season'):
    """
    Calculate the 25th, 50th, and 75th percentiles for specified columns grouped by the 'Season' column.
    """
    thresholds = {}
    for column in columns:
        thresholds[column] = df.groupby(group_column)[column].quantile([0.25, 0.50, 0.75]).unstack()
    return thresholds


def classify_uhi(row, temp_thresholds, humidity_thresholds, wind_speed_thresholds):
    """
    Classifies UHI intensity based on thresholds for temperature, humidity, and wind speed.
    """
    season = row['Season']
    temp = row['HourlyDryBulbTemperature']
    humidity = row['HourlyRelativeHumidity']
    wind_speed = row['HourlyWindSpeed']

    # Get thresholds for the current season
    temp_high = temp_thresholds.loc[season, 0.50]
    temp_medium = temp_thresholds.loc[season, 0.25]
    humidity_low = humidity_thresholds.loc[season, 0.25]
    wind_speed_low = wind_speed_thresholds.loc[season, 0.25]

    # Classify UHI intensity
    if temp > temp_high and humidity < humidity_low and wind_speed < wind_speed_low:
        return 'High'
    elif temp > temp_medium:
        return 'Medium'
    else:
        return 'Low'


def apply_uhi_classification(df, thresholds):
    """
    Applies UHI classification to a dataset using thresholds for temperature, humidity, and wind speed.
    """
    temp_thresholds = thresholds['HourlyDryBulbTemperature']
    humidity_thresholds = thresholds['HourlyRelativeHumidity']
    wind_speed_thresholds = thresholds['HourlyWindSpeed']

    df['UHI Intensity'] = df.apply(
        lambda row: classify_uhi(row, temp_thresholds, humidity_thresholds, wind_speed_thresholds), axis=1
    )
    return df
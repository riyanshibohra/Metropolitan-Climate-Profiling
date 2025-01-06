import pandas as pd

def calculate_seasonal_statistics(df, group_by_column, target_columns):
    """
    Calculates seasonal statistics (e.g., mean, median, quantiles) for specified columns.

    Parameters:
        df (pd.DataFrame): The dataset.
        group_by_column (str): The column to group by (e.g., 'Season').
        target_columns (list): List of columns to calculate statistics for.

    Returns:
        dict: A dictionary containing DataFrames of calculated statistics for each target column.
    """
    seasonal_stats = {}
    for column in target_columns:
        stats = df.groupby(group_by_column)[column].describe()
        seasonal_stats[column] = stats
        print(f"Seasonal statistics for {column}:\n{stats}\n")
    return seasonal_stats


def classify_uhi_intensity(row, temp_thresholds, humidity_thresholds, wind_speed_thresholds):
    """
    Classifies UHI intensity into 'High', 'Medium', or 'Low' based on thresholds.

    Parameters:
        row (pd.Series): A single row of the DataFrame.
        temp_thresholds (pd.DataFrame): Temperature thresholds by season.
        humidity_thresholds (pd.DataFrame): Humidity thresholds by season.
        wind_speed_thresholds (pd.DataFrame): Wind speed thresholds by season.

    Returns:
        str: The UHI intensity classification ('High', 'Medium', 'Low').
    """
    season = row['Season']
    temp = row['HourlyDryBulbTemperature']
    humidity = row['HourlyRelativeHumidity']
    wind_speed = row['HourlyWindSpeed']

    temp_high = temp_thresholds.loc[season, 0.50]
    temp_medium = temp_thresholds.loc[season, 0.25]
    humidity_low = humidity_thresholds.loc[season, 0.25]
    wind_speed_low = wind_speed_thresholds.loc[season, 0.25]

    if temp > temp_high and humidity < humidity_low and wind_speed < wind_speed_low:
        return 'High'
    elif temp > temp_medium:
        return 'Medium'
    else:
        return 'Low'


def apply_uhi_classification(df, temp_thresholds, humidity_thresholds, wind_speed_thresholds):
    """
    Applies UHI intensity classification to the dataset.

    Parameters:
        df (pd.DataFrame): The dataset.
        temp_thresholds (pd.DataFrame): Temperature thresholds by season.
        humidity_thresholds (pd.DataFrame): Humidity thresholds by season.
        wind_speed_thresholds (pd.DataFrame): Wind speed thresholds by season.

    Returns:
        pd.DataFrame: The dataset with an added 'UHI Intensity' column.
    """
    df['UHI Intensity'] = df.apply(
        lambda row: classify_uhi_intensity(row, temp_thresholds, humidity_thresholds, wind_speed_thresholds),
        axis=1
    )
    print(f"UHI Intensity classification applied. Counts:\n{df['UHI Intensity'].value_counts()}")
    return df
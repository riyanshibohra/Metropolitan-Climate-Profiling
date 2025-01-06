import pandas as pd
from sklearn.impute import KNNImputer

def impute_missing_values(df, cols_to_impute, n_neighbors=5):
    """
    Imputes missing values in specified columns using KNN imputation.

    Parameters:
        df (pd.DataFrame): The dataset.
        cols_to_impute (list): List of columns to impute.
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: The dataset with missing values imputed.
    """
    print(f"Missing values before imputation:\n{df[cols_to_impute].isnull().sum()}")
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    print(f"Missing values after imputation:\n{df[cols_to_impute].isnull().sum()}")
    return df


def remove_outliers(df, column):
    """
    Removes outliers from a specific column using the IQR method.

    Parameters:
        df (pd.DataFrame): The dataset.
        column (str): The column to remove outliers from.

    Returns:
        pd.DataFrame: The dataset with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    original_shape = df.shape
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print(f"Outliers removed from {column}. Original shape: {original_shape}, New shape: {df.shape}")
    return df


def extract_time_features(df, date_column):
    """
    Extracts time-based features from a datetime column.

    Parameters:
        df (pd.DataFrame): The dataset.
        date_column (str): The name of the datetime column.

    Returns:
        pd.DataFrame: The dataset with extracted time features.
    """
    df['Hour'] = df[date_column].dt.hour
    df['DayOfWeek'] = df[date_column].dt.dayofweek
    df['Month'] = df[date_column].dt.month
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)  # Calculate seasons
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['Season'] = df['Season'].map(seasons)

    print(f"Extracted time features from {date_column}")
    return df
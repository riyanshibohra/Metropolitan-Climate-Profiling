import pandas as pd
from sklearn.impute import KNNImputer

def convert_to_datetime(df, date_column):
    """
    Converts a specified column to datetime format.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df

def filter_columns(df, columns_of_interest):
    """
    Filters a DataFrame to retain only specified columns.
    """
    return df[columns_of_interest]

def convert_numeric_columns(df, columns):
    """
    Converts specified columns to numeric, coercing errors to NaN.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_missing_values(df, columns_to_impute, n_neighbors=5):
    """
    Imputes missing values in specified columns using KNN Imputer.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df

def extract_time_features(df, date_column):
    """
    Extracts time-based features (hour, month, season) from a datetime column.
    """
    df['Hour'] = df[date_column].dt.hour
    df['Month'] = df[date_column].dt.month
    df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['Season'] = df['Season'].map(seasons)
    return df

def remove_outliers(df, column, iqr_multiplier=1.5):
    """
    Removes outliers from a column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


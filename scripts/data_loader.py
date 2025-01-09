import pandas as pd
import os

def get_project_root():
    """Get the path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_dataset(city="Dallas"):
    """
    Load a city's climate data.
    
    Args:
        city (str): Name of the city (Dallas, Arlington, or Denton)
    """
    data_path = os.path.join(get_project_root(), 'data', f'{city}.csv')
    return pd.read_csv(data_path)

def load_raw_dataset(file_path, columns_of_interest=None, date_column=None):
    df = pd.read_csv(file_path)
    if columns_of_interest:
        df = df[columns_of_interest]
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
    return df

def load_processed_dataset(file_path):
    """
    Load a processed dataset from a file.
    
    Args:
        file_path (str or Path): Path to the dataset file
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Convert Path object to string if necessary
    file_path = str(file_path)
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        return pd.read_pickle(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv, .pkl, or .parquet.")

def save_dataset(df, file_path):
    """
    Saves a DataFrame to a file. Supports .csv, .pkl, and .parquet formats.
    """
    file_path = str(file_path)  # Convert Path object to string
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False)
    elif file_path.endswith('.pkl'):
        df.to_pickle(file_path)
    elif file_path.endswith('.parquet'):
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .csv, .pkl, or .parquet.")
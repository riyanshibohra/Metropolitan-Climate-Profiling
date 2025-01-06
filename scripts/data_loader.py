import pandas as pd

def load_dataset(url, columns_of_interest=None, date_column=None):
    """
    Loads a dataset from a given URL.

    Parameters:
        url (str): The URL to the dataset.
        columns_of_interest (list, optional): List of columns to keep. Default is None (keeps all columns).
        date_column (str, optional): Name of the column to parse as datetime. Default is None.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Load the dataset
    df = pd.read_csv(url, header='infer', low_memory=False)
    
    # Convert the date column to datetime
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])

    # Select columns of interest
    if columns_of_interest:
        df = df[columns_of_interest]

    print(f"Loaded dataset from {url}")
    print(f"Shape of the dataset: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    return df
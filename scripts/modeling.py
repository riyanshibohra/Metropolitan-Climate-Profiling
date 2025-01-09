import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Data preparation
def prepare_data(df, target_column='UHI Intensity'):
    """
    Prepares the dataset for modeling by splitting into features and target and encoding the target.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    # Make a copy to avoid modifying the original data
    df = df.copy()
    
    # Convert all numeric columns to float
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    # Drop any non-numeric columns except target and Season
    non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    cols_to_drop = [col for col in non_numeric_cols if col not in [target_column, 'Season']]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Prepare features and target
    X = df.drop(columns=[target_column, 'Season'])
    y = df[target_column]
    
    # Print data info for debugging
    print("Features shape:", X.shape)
    print("Features dtypes:\n", X.dtypes)
    print("Target unique values:", y.unique())
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.3, 
        random_state=42,
        stratify=y_encoded  # Ensure balanced split
    )
    
    return X_train, X_test, y_train, y_test, label_encoder

# Train a model
def train_model(model, X_train, y_train):
    """
    Trains a machine learning model.
    """
    model.fit(X_train, y_train)
    return model

# Evaluate a model
def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the model on the test set and prints classification metrics.
    """
    y_pred = model.predict(X_test)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred_decoded))

# Save the best model
def save_model(model, file_path):
    """
    Saves the trained model to a file.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def split_data(df, feature_columns, target_column, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        df (pd.DataFrame): The dataset.
        feature_columns (list): List of feature columns.
        target_column (str): Target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Trains a model and evaluates it on the test set.

    Parameters:
        model: The model to train.
        X_train, X_test, y_train, y_test: Training and testing data.
        model_name (str): Name of the model (for print statements).

    Returns:
        model: The trained model.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return model


def compare_models(models, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates multiple models, comparing their performance.

    Parameters:
        models (dict): Dictionary of models with model names as keys.
        X_train, X_test, y_train, y_test: Training and testing data.

    Returns:
        dict: Dictionary of trained models.
    """
    trained_models = {}
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        trained_models[model_name] = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    return trained_models
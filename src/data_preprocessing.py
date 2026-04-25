"""
Data Preprocessing Module for Healthcare Readmission Prediction.

Provides functions for loading and splitting healthcare datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load healthcare dataset from CSV file.

    Args:
        path: Path to the CSV file containing healthcare data.

    Returns:
        pd.DataFrame: Loaded dataframe with healthcare records.
    """
    df = pd.read_csv(path)
    return df


def split_data(df):
    """
    Split dataframe into training and test sets.

    Args:
        df: DataFrame containing healthcare data with 'readmitted' target column.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Features and target split.
    """
    # Separate features and target variable
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    # Split with 80/20 ratio, stratified by target for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

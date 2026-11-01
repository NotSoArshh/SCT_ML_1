"""
House Price Prediction using Linear Regression
==============================================
This script trains a Multiple Linear Regression model using scikit-learn to predict
house sale prices based on key features: square footage, number of bedrooms, and bathrooms.

Dataset expected: Kaggle's House Prices - Advanced Regression Techniques dataset (train.csv).

Usage:
    python house_price_prediction.py --data train.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_preprocess(filepath: str):
    """
    Load dataset and select relevant features for linear regression.
    
    Parameters:
        filepath (str): Path to the CSV dataset.
        
    Returns:
        X (pd.DataFrame): Feature matrix (Square Footage, Bedrooms, Bathrooms)
        y (pd.Series): Target variable (SalePrice)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found. Please provide a valid file path.")
    
    df = pd.read_csv(filepath)
    
    # Standard column mapping for Kaggle House Prices dataset
    # GrLivArea: Above grade (ground) living area square feet
    # BedroomAbvGr: Bedrooms above grade
    # FullBath: Full bathrooms above grade
    target_col = "SalePrice"
    feature_cols = ["GrLivArea", "BedroomAbvGr", "FullBath"]
    
    # Check if necessary columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    # Clean dataset by dropping missing values in selected features/target
    df_clean = df[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Train Linear Regression model and evaluate performace.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("      LINEAR REGRESSION MODEL METRICS     ")
    print("="*40)
    print(f"MAE  (Mean Absolute Error)     : ${mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error) : ${rmse:,.2f}")
    print(f"R² Score                        : {r2:.4f}")
    print("="*40)
    
    print("\nModel Formula / Coefficients:")
    print(f"Intercept: ${model.intercept_:,.2f}")
    for feature, coef in zip(X.columns, model.coef_):
        print(f" - {feature:<15}: ${coef:,.2f} per unit")
    print("="*40 + "\n")
    
    return model


def predict_sample(model: LinearRegression):
    """
    Run sample predictions on custom input values.
    """
    samples = pd.DataFrame([
        {"GrLivArea": 1500, "BedroomAbvGr": 3, "FullBath": 2},
        {"GrLivArea": 2500, "BedroomAbvGr": 4, "FullBath": 3},
        {"GrLivArea": 1000, "BedroomAbvGr": 2, "FullBath": 1},
    ])
    
    print("Sample Predictions:")
    predictions = model.predict(samples)
    for idx, row in samples.iterrows():
        print(f"House {idx+1} ({int(row['GrLivArea'])} sqft, {int(row['BedroomAbvGr'])} beds, {int(row['FullBath'])} baths): "
              f"Estimated Price = ${predictions[idx]:,.2f}")


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction using Linear Regression")
    parser.add_argument("--data", type=str, default="train.csv", help="Path to the train.csv data file")
    args = parser.parse_args()
    
    try:
        X, y = load_and_preprocess(args.data)
        model = train_and_evaluate(X, y)
        predict_sample(model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
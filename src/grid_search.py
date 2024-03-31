import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def param_search(X_train, X_test, y_train, y_test):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['auto', 'sqrt'],
        'class_weight': ['balanced'],
        'bootstrap': [True]
    }

    # Initialize Random Forest classifier
    rf = RandomForestClassifier()

    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', refit='roc_auc', verbose=5)
    grid_search.fit(X_train, y_train)

    # Extract CV results
    cv_results = grid_search.cv_results_ 

if __name__ == "__main__":
    df = pd.read_csv(config.SMOTE_DATA_FILES)
    X = df.drop('status', axis=1)
    y = df['status']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform grid search and plot CV results
    param_search(X_train, X_test, y_train, y_test)
    

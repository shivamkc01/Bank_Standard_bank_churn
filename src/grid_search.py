    import config
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from xgboost import XGBClassifier

    def param_search(X_train, X_test, y_train, y_test):
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 150, 200, 300],
            'max_depth': [1, 2,3],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1],
            'scale_pos_weight': [1, 2, 3, 4, 5]
        }

        # Initialize Random Forest classifier
        rf = XGBClassifier()

        # Perform random search with 5-fold cross-validation
        random_search = RandomizedSearchCV(estimator=rf, 
                                        param_distributions=param_grid, 
                                        cv=10, scoring='roc_auc', 
                                        refit=True, verbose=5, 
                                        return_train_score=True)
        random_search.fit(X_train, y_train)

        # Print best parameters and score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Extract CV results
        cv_results = random_search.cv_results_

        # Plot training and testing scores for every fold
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 6), cv_results['mean_train_score'], label='Mean Train Score', marker='o')
        plt.plot(range(1, 6), cv_results['mean_test_score'], label='Mean Test Score', marker='o')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Training and Testing Scores for Each Fold')
        plt.xticks(range(1, 6))
        plt.legend()
        plt.grid(True)
        plt.show()

    if __name__ == "__main__":
        df = pd.read_csv(config.SMOTE_DATA_FILES)
        X = df.drop('status', axis=1)
        y = df['status']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform random search and plot CV results
        param_search(X_train, X_test, y_train, y_test)

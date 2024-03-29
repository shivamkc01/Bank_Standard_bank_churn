import os
import random 
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import config
from sklearn.metrics import roc_curve, auc

def seed_it_all(seed=config.SEED):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def feature_with_date(df):
    # Calculate additional features
    current_date = datetime.datetime.now()
    df['Age'] = (current_date - df['CUS_DOB']).dt.days / 365.25  # Age in years
    df['Tenure'] = (current_date - df['CUS_Customer_Since']).dt.days / 365.25  # Tenure in years
    df['Customer_Age_At_Onboarding'] = (df['CUS_Customer_Since'] - df['CUS_DOB']).dt.days / 365.25  # Age at onboarding

    # Convert features to int or float data types
    df['Age'] = df['Age'].astype(float)
    df['Tenure'] = df['Tenure'].astype(float)
    df['Customer_Age_At_Onboarding'] = df['Customer_Age_At_Onboarding'].astype(float)

    # Display the DataFrame with additional features
    return df 


def label_encode_categorical(df, cols):
    """_summary_
    Args:
        df (dataframe): Input DataFrame with categorical columns.
        cols (list): List of column names to be label encoded.
    """
    df_encoded = df.copy()

    for col in cols:
        le = preprocessing.LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])

    return df_encoded

def remove_outliers_iqr(df, columns, threshold=1.5):
    """
    Remove outliers from a DataFrame using the Interquartile Range (IQR) method.
    Args:
        df (DataFrame): Input DataFrame.
        columns (list): List of column names to detect outliers.
        threshold (float): Threshold value to determine outliers. Default is 1.5.

    Returns:
        DataFrame: DataFrame with outliers removed.
    """
    print(f"Before outliers removing shape of data: {df.shape}")
    outlier_indices = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_indices.extend(outliers.index)
    df = df.drop(outlier_indices)
    print(f"After outliers removing shape of data: {df.shape}")
    return df


def plot_histogram(df):
    # Determine the number of rows and columns for subplots
    num_features = len(df.columns)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    # Set the size of the figure
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3*num_rows))

    # Iterate through each column in the DataFrame
    for i, col in enumerate(df.columns):
        row_idx = i // num_cols
        col_idx = i % num_cols
        
        sns.kdeplot(df[col], shade=True, ax=axes[row_idx, col_idx], color="#A0153E", alpha=0.5)
        
        axes[row_idx, col_idx].set_title(col)
        # Remove y-axis lines for all subplots except the bottom row
        if row_idx < num_rows - 1:
            axes[row_idx, col_idx].tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
            axes[row_idx, col_idx].spines['left'].set_visible(False)
            axes[row_idx, col_idx].spines['right'].set_visible(False)
            axes[row_idx, col_idx].spines['top'].set_visible(False)
            axes[row_idx, col_idx].spines['bottom'].set_visible(False)

    # Remove any empty subplots
    for i in range(num_features, num_rows*num_cols):
        fig.delaxes(axes.flatten()[i])
    

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(X_before_scaling, X_after_scaling):
    num_features = X_before_scaling.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    for i in range(num_features):
        # Plot distribution before scaling
        sns.histplot(X_before_scaling[:, i], ax=axes[0], kde=True, color='blue', alpha=0.5, label='Before Scaling', stat='density')
        
        # Plot distribution after scaling
        sns.histplot(X_after_scaling[:, i], ax=axes[1], kde=True, color='red', alpha=0.5, label='After Scaling', stat='density')
    
    # Set titles and labels
    axes[0].set_title('Feature Distributions Before Scaling')
    axes[0].legend()
    axes[1].set_title('Feature Distributions After Scaling')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve_for_classes(clf, x, y, class_labels, title):
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(class_labels):
        y_binary = (y == label).astype(int)
        y_score = clf.predict_proba(x)[:, i]
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve - Class {label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"../plots/roc_auc_curve/folds_wihtout_sampling/{title}.jpg" ,dpi=100)
    plt.close()
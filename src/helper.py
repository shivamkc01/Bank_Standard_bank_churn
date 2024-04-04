import os
import random 
import pandas as pd
import itertools
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import config
from sklearn.metrics import roc_curve, auc, confusion_matrix

def seed_it_all(seed=config.SEED):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_features(data):
    data['high_monthly_income'] = (data['cus_month_income'] > data['cus_month_income'].quantile(0.75)).astype(int)
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])
    data = pd.get_dummies(data, columns=['age_group'])
    data = pd.get_dummies(data, columns=['cus_gender', 'cus_marital_status'])
    data['income_category'] = pd.qcut(data['cus_month_income'], q=3, labels=['Low', 'Medium', 'High'])
    data = pd.get_dummies(data, columns=['income_category'])
    data['total_transactions_last_3_months'] = data['total_transactions'] - data['total_transactions'].shift(3, fill_value=0)
    data['credit_to_debit_ratio'] = data['total_credit_amt'] / (data['total_debit_amt'] + 1)  # Add 1 to avoid division by zero
    data['customer_tenure'] = 2024 - data['cus_customer_since']
    data['credit_utilization'] = data['total_credit_amt'] / (data['total_credit_amt'] + data['total_debit_amt'] + 1)  # Add 1 to avoid division by zero
    data['debit_to_credit_transaction_ratio'] = data['total_debit_trans'] / (data['total_credit_trans'] + 1)  # Add 1 to avoid division by zero
    data['interaction_age_income'] = data['age'] * data['cus_month_income']


    data['transaction_velocity'] = data['total_transactions'] / (data['years_with_us'] + 1)  # Add 1 to avoid division by zero
    data['credit_utilization_ratio'] = data['total_credit_amt'] / (data['total_credit_amt'] + data['total_debit_amt'] + 1)  # Add 1 to avoid division by zero    data['income_stability'] = data['cus_month_income'].rolling(window=3).std() / data['cus_month_income'].rolling(window=3).mean()
    data['income_stability'] = data['cus_month_income'].rolling(window=3).std() / (data['cus_month_income'].rolling(window=3).mean() + 1)  # Add 1 to avoid division by zero

    data['customer_tenure'] = 2024 - data['cus_customer_since']
    data['debt_to_income_ratio'] = data['total_debit_amt'] / (data['cus_month_income'] + 1)  # Add 1 to avoid division by zero
    

    return data


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
    plt.savefig(f"../plots/roc_auc_curve/{title}.jpg" ,dpi=100)
    plt.close()



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title if title else 'Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"../results/confusion_matrix/{title}.jpg" ,dpi=100)
    plt.close()
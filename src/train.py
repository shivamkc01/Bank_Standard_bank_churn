import pandas as pd
import numpy as np 

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import tree 
from sklearn import metrics
from colorama import Fore 
from tqdm import tqdm
import config
import data_cleaning
from helper import plot_roc_curve_for_classes

train_roc_auc_list = []
test_roc_auc_list = []
train_classification_reports = []
test_classification_reports = []
def training(fold):
    df = pd.read_csv("../data/fold_data/df_folds.csv")
    df_train = df[df.kfold != fold].reset_index()
    df_valid = df[df.kfold == fold].reset_index()

    combined_df = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)

    # Encode categorical columns
    label_encoder = preprocessing.LabelEncoder()
    for column in combined_df.select_dtypes(include=['object']).columns:
        combined_df[column] = label_encoder.fit_transform(combined_df[column])
    
    # Split back into training and validation sets
    train_enc = combined_df[:len(df_train)]
    test_enc = combined_df[len(df_train):]
    xtrain = train_enc.drop('status', axis=1).values
    ytrain = train_enc.status.values 

    xvalid = test_enc.drop('status', axis=1).values
    yvalid = test_enc.status.values

    scaler = preprocessing.StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xvalid = scaler.transform(xvalid)

    model = linear_model.LogisticRegression(class_weight={0:1, 1:4})
    model.fit(xtrain, ytrain)
    train_preds_prob = model.predict_proba(xtrain)[:,1]
    test_preds_prob = model.predict_proba(xvalid)[:,1]

    train_roc_auc = metrics.roc_auc_score(ytrain, train_preds_prob)
    test_roc_auc = metrics.roc_auc_score(yvalid, test_preds_prob)
    train_classification_report = metrics.classification_report(
        ytrain, 
        model.predict(xtrain), 
        target_names=['Not_churn_cus', 'churn_cus']
        )
    print(f"TRAINING SCORE: \n{train_classification_report}")
    test_classification_report = metrics.classification_report(
        yvalid, 
        model.predict(xvalid), 
        target_names=['Not_churn_cus', 'churn_cus']
        )
    print(f"TESTING SCORE: \n{test_classification_report}")
    print(Fore.GREEN+f"Train ROC AUC: {train_roc_auc}")
    print(Fore.GREEN+f"Test ROC AUC: {test_roc_auc}")
    
    train_classification_reports.append(train_classification_report)
    test_classification_reports.append(test_classification_report)

    plot_roc_curve_for_classes(model, xtrain, ytrain, [0, 1], f'Training ROC Curve for Fold {fold+1}')
    plot_roc_curve_for_classes(model, xvalid, yvalid, [0, 1], f'Testing ROC Curve for Fold {fold+1}')

    return train_roc_auc, test_roc_auc, model
if __name__ == "__main__":

    train_roc_auc_avg = []
    test_roc_auc_avg = []
    for fold in tqdm(range(10)):
        train_roc_auc, test_roc_auc, model = training(fold)
        train_roc_auc_avg.append(train_roc_auc)
        test_roc_auc_avg.append(test_roc_auc)

    print(f"Overall Training ROC Score: {np.mean(train_roc_auc_avg)}, Testing ROC Score : {np.mean(test_roc_auc_avg)}")
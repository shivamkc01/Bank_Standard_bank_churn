import os
import argparse
import logging
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import tree 
from sklearn import metrics
from colorama import Fore 
from tqdm import tqdm
import config
import model_dispatcher
import data_cleaning
from helper import plot_roc_curve_for_classes

train_roc_auc_list = []
test_roc_auc_list = []
train_classification_reports = []
test_classification_reports = []

def training(fold, model, encoding=False):
    print("#"*20)
    print(f"### FOLD {fold} ###")
    logging.info(f"### FOLD {fold} ###", )
    df = pd.read_csv(config.SMOTE_FOLD_FILES)
    # df.drop('cus_dob', axis=1, inplace=True)
    # print(df.head())
    df_train = df[df.kfold != fold].reset_index()
    df_valid = df[df.kfold == fold].reset_index()

    combined_df = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)

    if encoding:
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

    # # Robust Scaling
    # robust_scaler = preprocessing.RobustScaler()
    # xtrain = robust_scaler.fit_transform(xtrain)
    # xvalid = robust_scaler.transform(xvalid)

    # Quantile Transformer
    quantile_transformer = preprocessing.QuantileTransformer()
    xtrain = quantile_transformer.fit_transform(xtrain)
    xvalid = quantile_transformer.transform(xvalid)


    model = model_dispatcher.models[model]
    logging.info(f"Model Name: {args.model}")
    logging.info(f"Hyperparameters: {model.get_params()}")
    model.fit(xtrain, ytrain)
    train_preds_prob = model.predict_proba(xtrain)[:,1]
    test_preds_prob = model.predict_proba(xvalid)[:,1]

    train_roc_auc = metrics.roc_auc_score(ytrain, train_preds_prob)
    logging.info(f"Traning ROC Score: {train_roc_auc}")
    test_roc_auc = metrics.roc_auc_score(yvalid, test_preds_prob)
    logging.info(f"Testing ROC Score: {test_roc_auc}")
    train_classification_report = metrics.classification_report(
        ytrain, 
        model.predict(xtrain), 
        target_names=['Not_churn_cus', 'churn_cus']
        )
    logging.info(f"Classification Report - Traning (Fold {fold}):\n{train_classification_report}")
    test_classification_report = metrics.classification_report(
        yvalid, 
        model.predict(xvalid), 
        target_names=['Not_churn_cus', 'churn_cus']
        )
    logging.info(f"Classification Report - Testing (Fold {fold}):\n{test_classification_report}")
    logging.info("#"*30)
    print(Fore.GREEN+f"Train ROC AUC: {train_roc_auc}")
    print(Fore.GREEN+f"Test ROC AUC: {test_roc_auc}")
    
   
    plot_roc_curve_for_classes(model, xtrain, ytrain, [0, 1], f'Training ROC Curve for Fold {fold+1}')
    plot_roc_curve_for_classes(model, xvalid, yvalid, [0, 1], f'Testing ROC Curve for Fold {fold+1}')

    return train_roc_auc, test_roc_auc, model
if __name__ == "__main__":

    # Lets define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "dt", "rf", "xgb"])
    parser.add_argument("--logs", type=str)
    args = parser.parse_args()


    logging.basicConfig(filename=f"{config.LOGGING_FILS}/{args.logs}.log", level=logging.INFO)

    train_roc_auc_avg = []
    test_roc_auc_avg = []
    for fold in range(args.fold):
        train_roc_auc, test_roc_auc, model = training(fold, args.model)
        train_roc_auc_avg.append(train_roc_auc)
        test_roc_auc_avg.append(test_roc_auc)

    logging.info(f"Overall ROC score on all 10 FOLDs in TRAINING SET={np.mean(train_roc_auc_avg)}, TESTING SET={np.mean(test_roc_auc_avg)}") 
    print(f"Overall Training ROC Score: {np.mean(train_roc_auc_avg)}, Testing ROC Score : {np.mean(test_roc_auc_avg)}")
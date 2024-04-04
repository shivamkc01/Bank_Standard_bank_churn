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
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import config
import model_dispatcher
import data_cleaning
from helper import plot_roc_curve_for_classes, plot_confusion_matrix, create_features

train_roc_auc_list = []
test_roc_auc_list = []
train_classification_reports = []
test_classification_reports = []

def training(fold, model, encoding=False, plot_roc = False, plot_conf_matrix=False, metric='roc_auc'):
    print("#"*20)
    print(f"### FOLD {fold} ###")
    logging.info(f"### FOLD {fold} ###", )
    df = pd.read_csv(config.SMOTE_FOLD_FILES)
    # df.drop('cus_dob', axis=1, inplace=True)
    # print(df.head())
    df_train = df[df.kfold != fold].reset_index()
    df_valid = df[df.kfold == fold].reset_index()

    combined_df = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)

    combined_df = create_features(combined_df)
    if encoding:
        # Encode categorical columns
        label_encoder = preprocessing.LabelEncoder()
        for column in combined_df.select_dtypes(include=['object']).columns:
            combined_df[column] = label_encoder.fit_transform(combined_df[column])
    
    # Replace NaN values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    combined_df = pd.DataFrame(imputer.fit_transform(combined_df), columns=combined_df.columns)
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
    train_preds = model.predict(xtrain)
    test_preds = model.predict(xvalid)
    train_preds_prob = model.predict_proba(xtrain)[:,1]
    test_preds_prob = model.predict_proba(xvalid)[:,1]

    train_scores = {}
    test_scores = {}
    
    if metric == 'f1_score':
        train_score = metrics.f1_score(ytrain, train_preds)
        test_score = metrics.f1_score(yvalid, test_preds)
    elif metric == 'roc_auc':
        train_score = metrics.roc_auc_score(ytrain, train_preds_prob)
        test_score = metrics.roc_auc_score(yvalid, test_preds_prob)
    elif metric == 'precision':
        train_score = metrics.precision_score(ytrain, train_preds)
        test_score = metrics.precision_score(yvalid, test_preds)
    elif metric == 'recall':
        train_score = metrics.recall_score(ytrain, train_preds)
        test_score = metrics.recall_score(yvalid, test_preds)
    elif metric == 'accuracy':
        train_score = metrics.accuracy_score(ytrain, train_preds)
        test_score = metrics.accuracy_score(yvalid, test_preds)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # train_scores[metric] = train_score
    # test_scores[metric] = test_score

    print(f"Training {metric}: {train_score}")
    print(f"Testing {metric}: {test_score}")

    # train_roc_auc = metrics.roc_auc_score(ytrain, train_preds_prob)
    # logging.info(f"Traning ROC Score: {train_roc_auc}")
    # test_roc_auc = metrics.roc_auc_score(yvalid, test_preds_prob)
    # logging.info(f"Testing ROC Score: {test_roc_auc}")
    # train_classification_report = metrics.classification_report(
    #     ytrain, 
    #     model.predict(xtrain), 
    #     target_names=['Not_churn_cus', 'churn_cus']
    #     )
    # logging.info(f"Classification Report - Traning (Fold {fold}):\n{train_classification_report}")
    # test_classification_report = metrics.classification_report(
    #     yvalid, 
    #     model.predict(xvalid), 
    #     target_names=['Not_churn_cus', 'churn_cus']
    #     )
    # logging.info(f"Classification Report - Testing (Fold {fold}):\n{test_classification_report}")
    # logging.info("#"*30)
    # print(Fore.GREEN+f"Train ROC AUC: {train_roc_auc}")
    # print(Fore.GREEN+f"Test ROC AUC: {test_roc_auc}")
    
    if plot_roc:
        plot_roc_curve_for_classes(model, xtrain, ytrain, [0, 1], f'Training ROC Curve for Fold {fold+1} using {args.model} ')
        plot_roc_curve_for_classes(model, xvalid, yvalid, [0, 1], f'Testing ROC Curve for Fold {fold+1} using {args.model} ')

    if plot_conf_matrix:
        plot_confusion_matrix(ytrain, train_preds, ['Not_churn_cus', 'churn_cus'], title=f'Training Confusion Matrix with {args.model}- Fold {fold+1}')
        plot_confusion_matrix(yvalid, test_preds, ['Not_churn_cus', 'churn_cus'], title=f'Testing Confusion Matrix with {args.model}- Fold {fold+1}')

    return train_score, test_score, model

if __name__ == "__main__":

    # Lets define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "dt", "rf", "xgb"])
    parser.add_argument("--logs", type=str, default=None)
    parser.add_argument("--confusion_matrix", type=bool, default=False)
    parser.add_argument("--metric", type=str, default='roc_auc', choices=['f1_score', 'roc_auc', 'precision', 'recall', 'accuracy'])
    parser.add_argument("--auc_plot", type=bool, default=False)
    args = parser.parse_args()


    logging.basicConfig(filename=f"{config.LOGGING_FILS}/{args.logs}.log", level=logging.INFO)

    train_roc_auc_avg = []
    test_roc_auc_avg = []
    for fold in range(args.fold):
        train_roc_auc, test_roc_auc, model = training(fold, args.model, plot_roc=args.auc_plot, plot_conf_matrix=args.confusion_matrix, metric=args.metric)
        train_roc_auc_avg.append(train_roc_auc)
        test_roc_auc_avg.append(test_roc_auc)

    logging.info(f"Overall {args.metric} on all 10 FOLDs in TRAINING SET={np.mean(train_roc_auc_avg)}, TESTING SET={np.mean(test_roc_auc_avg)}") 
    print(f"Overall Training {args.metric}: {np.mean(train_roc_auc_avg)}, Testing {args.metric} : {np.mean(test_roc_auc_avg)}")
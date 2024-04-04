import os
import argparse
import shap
import numpy as np
import pandas as pd
from sklearn import preprocessing
import model_dispatcher
import config
import helper
import data_cleaning
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def interpret_model(model, data, feature_names,save_path):
    plt.tight_layout()
    explainer = shap.Explainer(model)
    
    shap_values = explainer.shap_values(data)
    
    summary_plot_path = os.path.join(save_path, "summary_plot.png")
    shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    
    waterfall_plot_path = os.path.join(save_path, "waterfall_plot.png")
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=data.iloc[0]), max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_plot_path)
    plt.close()
    
    bar_plot_path = os.path.join(save_path, "bar_plot.png")
    shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type='bar', show=False)    
    plt.tight_layout()
    plt.savefig(bar_plot_path)
    plt.close()   

    top_5_indices = np.argsort(np.mean(np.abs(shap_values), axis=0))[-5:]
    for idx in top_5_indices:
        dependence_plot_path = os.path.join(save_path, f"dependence_plot_{idx}.png")
        shap.dependence_plot(idx, shap_values, data, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(dependence_plot_path)
        plt.close()


def main():

    df = pd.read_csv(config.SMOTE_FOLD_FILES)
    combined_df = helper.create_features(df)
    label_encoder = preprocessing.LabelEncoder()
    for column in combined_df.select_dtypes(include=['object']).columns:
        combined_df[column] = label_encoder.fit_transform(combined_df[column])
    imputer = SimpleImputer(strategy='mean')
    combined_df = pd.DataFrame(imputer.fit_transform(combined_df), columns=combined_df.columns)
    
    X = combined_df.drop('status', axis=1)
    y = combined_df['status']
    
    model = model_dispatcher.models['xgb']
    
    model.fit(X, y)
    
    interpret_model(model, X, feature_names=X.columns.tolist(), save_path="../results/interpert_plots/")

if __name__ == "__main__":
    main()

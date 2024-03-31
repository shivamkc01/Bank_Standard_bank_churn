import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import config
from sklearn import preprocessing
from model_dispatcher import models

def interpret_model(model_name, sample_index=0):
    # Load the data
    df = pd.read_csv(config.SMOTE_FOLD_FILES)
    combined_df = df.drop(columns=['status', 'kfold'])

    # Save feature names
    feature_names = combined_df.columns.tolist()

    # Encode categorical columns
    label_encoder = preprocessing.LabelEncoder()
    for column in combined_df.select_dtypes(include=['object']).columns:
        combined_df[column] = label_encoder.fit_transform(combined_df[column])

    # Scale the data
    scaler = preprocessing.StandardScaler()
    combined_df_scaled = scaler.fit_transform(combined_df)

    # Load the model
    model = models[model_name]

    # Train the model
    model.fit(combined_df_scaled, df['status'].values)

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, combined_df_scaled,check_additivity=False)

    # Calculate SHAP values
    shap_values = explainer(combined_df_scaled)

    # Get SHAP values for a specific instance
    shap_values_instance = shap_values[sample_index, :]

    # Extract feature names and SHAP values
    feature_names_instance = combined_df.columns.tolist()
    shap_values_instance = shap_values_instance.values

    # Create a waterfall plot
    shap.waterfall_plot(shap.Explanation(values=shap_values_instance,
                                          base_values=explainer.expected_value,
                                          data=combined_df.iloc[sample_index, :]),
                        max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(f"{config.SHAP_DIR}/{model_name}_waterfall_plot_sample_{sample_index}.png")
    plt.close()

    print("Waterfall plot created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "dt", "rf"])
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()

    interpret_model(args.model, args.sample_index)

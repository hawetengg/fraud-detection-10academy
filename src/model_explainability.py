import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Define base directories relative to this script's location
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

PROCESSED_DATA_PATH = os.path.join(project_root, 'data', 'processed')
MODELS_PATH = os.path.join(project_root, 'models')
REPORTS_FIGURES_PATH = os.path.join(project_root, 'reports', 'figures')
REPORTS_PATH = os.path.join(project_root, 'reports')

# Ensure output directories exist
os.makedirs(REPORTS_FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)
print(f"Ensured reports figures directory exists at: {REPORTS_FIGURES_PATH}")
print(f"Ensured models directory exists at: {MODELS_PATH}")
print(f"Ensured reports directory exists at: {REPORTS_PATH}")


# Load preprocessed datasets and trained models
print("\n--- Loading transformed datasets and trained models ---")
try:
    fraud_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'Fraud_Data_transformed.csv'))
    creditcard_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_cleaned.csv'))

    with open(os.path.join(MODELS_PATH, 'Fraud_Data_lgb_model.pkl'), 'rb') as f:
        fraud_lgb_model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'CreditCard_lgb_model.pkl'), 'rb') as f:
        credit_lgb_model = pickle.load(f)
    print("All data and models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: One or more required files not found. Please ensure data_preprocessing.py, feature_engineering.py, and model_training.py have been run successfully.")
    print(f"Missing file: {e}")
    exit()


# Prepare data (same split as Task 2 for consistency)
def prepare_data(data, target_col, test_size=0.2, random_state=42):
    cols_to_drop = ['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time']
    if target_col == 'Class' and 'Time' in data.columns:
        data = data.drop(columns=['Time'], errors='ignore')

    actual_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    
    X = data.drop(columns=[target_col] + actual_cols_to_drop, errors='ignore')
    y = data[target_col]

    _, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_test

# Get test data
print("\n--- Preparing test data for explainability ---")
fraud_X_test = prepare_data(fraud_data, 'class')
print(f"Fraud X_test shape: {fraud_X_test.shape}")

credit_X_test = prepare_data(creditcard_data, 'Class')
print(f"Credit Card X_test shape: {credit_X_test.shape}")


# SHAP Explainer for LightGBM
def generate_shap_plots(model, X_test, dataset_name, reports_figures_path):
    print(f"\nGenerating SHAP plots for {dataset_name}...")
    
    explainer = shap.TreeExplainer(model)
    
    # Check if shap_values is a list (for multi-output/binary classification)
    # and select the positive class (index 1) if it is.
    # Otherwise, assume it's a single array (e.g., regression or already selected class).
    shap_values_raw = explainer.shap_values(X_test)
    
    if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
        shap_values = shap_values_raw[1] # For binary classification, typically index 1 is the positive class
        expected_value = explainer.expected_value[1]
    else:
        shap_values = shap_values_raw
        expected_value = explainer.expected_value

    # Summary Plot (Global Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary Plot - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_figures_path, f'shap_summary_{dataset_name.lower()}.png'))
    plt.close()
    print(f"Saved SHAP Summary Plot: shap_summary_{dataset_name.lower()}.png")
    
    # Force Plot for first test instance (Local Feature Importance)
    if not X_test.empty:
        plt.figure(figsize=(12, 6))
        # Ensure we're using the correct shap_values slice for the first instance
        # For binary classification, shap_values is (num_samples, num_features)
        shap.force_plot(expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot (First Instance) - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_figures_path, f'shap_force_{dataset_name.lower()}.png'))
        plt.close()
        print(f"Saved SHAP Force Plot: shap_force_{dataset_name.lower()}.png")
    else:
        print(f"Skipping Force Plot for {dataset_name}: X_test is empty.")
    
    return shap_values

# Generate SHAP plots for both datasets
fraud_shap_values = generate_shap_plots(fraud_lgb_model, fraud_X_test, 'Fraud_Data', REPORTS_FIGURES_PATH)
credit_shap_values = generate_shap_plots(credit_lgb_model, credit_X_test, 'CreditCard', REPORTS_FIGURES_PATH)

# Save SHAP values for further analysis if needed
print("\n--- Saving SHAP values ---")
np.save(os.path.join(REPORTS_PATH, 'shap_values_fraud.npy'), fraud_shap_values)
print("Saved: shap_values_fraud.npy")
np.save(os.path.join(REPORTS_PATH, 'shap_values_creditcard.npy'), credit_shap_values)
print("Saved: shap_values_creditcard.npy")

print("\nModel Explainability script completed successfully.")
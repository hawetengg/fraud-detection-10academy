import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os # Import os for path handling

# Define base directory relative to this script's location
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
PROCESSED_DATA_PATH = os.path.join(project_root, 'data', 'processed')
REPORTS_FIGURES_PATH = os.path.join(project_root, 'reports', 'figures')
MODELS_PATH = os.path.join(project_root, 'models')

# Ensure output directories exist
os.makedirs(REPORTS_FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
print(f"Ensured reports figures directory exists at: {REPORTS_FIGURES_PATH}")
print(f"Ensured models directory exists at: {MODELS_PATH}")


# Load preprocessed datasets
print("\n--- Loading transformed datasets ---")
# Load the Fraud_Data_transformed.csv which is the output of feature_engineering.py
fraud_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'Fraud_Data_transformed.csv'))
# For creditcard_data, we assume it's still the cleaned version from data_preprocessing.py
creditcard_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_cleaned.csv'))

print(f"Loaded Fraud Data (transformed): {fraud_data.shape[0]} rows, {fraud_data.shape[1]} columns")
print(f"Loaded Credit Card Data (cleaned): {creditcard_data.shape[0]} rows, {creditcard_data.shape[1]} columns")


# Function to prepare data
def prepare_data(data, target_col, test_size=0.2, random_state=42):
    # Identify columns to drop from features (X)
    # These are typically identifiers or original datetime columns that have been engineered
    # 'class' or 'Class' is the target column and should not be in this list
    cols_to_drop = ['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time']
    
    # Filter out columns that don't exist in the current DataFrame
    actual_cols_to_drop = [col for col in cols_to_drop if col in data.columns]

    print(f"Dropping columns from features: {actual_cols_to_drop}")
    # Drop the target column and the identified non-feature columns
    X = data.drop(columns=[target_col] + actual_cols_to_drop, errors='ignore')
    y = data[target_col]

    print(f"Features (X) shape before split: {X.shape}")
    print(f"Target (y) shape before split: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"X_train shape before SMOTE: {X_train.shape}")
    print(f"y_train value counts before SMOTE:\n{y_train.value_counts()}")

    # Apply SMOTE to training data
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"X_train shape after SMOTE: {X_train_res.shape}")
    print(f"y_train value counts after SMOTE:\n{y_train_res.value_counts()}")
    
    return X_train_res, X_test, y_train_res, y_test

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    print(f"\n--- Training and Evaluating Models for {dataset_name} ---")
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    lr_model.fit(X_train, y_train)
    print("Logistic Regression training complete.")
    
    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
    lgb_model.fit(X_train, y_train)
    print("LightGBM training complete.")
    
    # Evaluate models
    models = {'Logistic Regression': lr_model, 'LightGBM': lgb_model}
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {'AUC-PR': auc_pr, 'F1-Score': f1, 'Confusion Matrix': cm}
        
        print(f"  AUC-PR: {auc_pr:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} ({dataset_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(REPORTS_FIGURES_PATH, f'cm_{name.lower().replace(" ", "_")}_{dataset_name.lower()}.png'))
        plt.close()
        print(f"Saved confusion matrix: cm_{name.lower().replace(' ', '_')}_{dataset_name.lower()}.png")
    
    # Save models
    print("\n--- Saving Models ---")
    with open(os.path.join(MODELS_PATH, f'{dataset_name}_lr_model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"Saved {dataset_name}_lr_model.pkl")

    with open(os.path.join(MODELS_PATH, f'{dataset_name}_lgb_model.pkl'), 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"Saved {dataset_name}_lgb_model.pkl")
    
    return results

# Prepare data for both datasets
print("\n--- Preparing Fraud Data ---")
fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test = prepare_data(fraud_data, 'class')

print("\n--- Preparing Credit Card Data ---")
# Note: creditcard_data has not gone through the full feature engineering pipeline (encoding, scaling)
# It only has numerical V features, Time, Amount, and Class.
# For a robust solution, creditcard_data should also go through a feature engineering script
# similar to fraud_data if it has categorical features or needs scaling.
# For now, we'll assume 'Time' is the only non-V-feature to consider dropping.
# 'Amount' is a feature, 'Class' is target.
credit_cols_to_drop = ['Time']
credit_X_train, credit_X_test, credit_y_train, credit_y_test = prepare_data(creditcard_data.drop(columns=credit_cols_to_drop, errors='ignore'), 'Class')


# Train and evaluate models
fraud_results = train_and_evaluate(fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test, 'Fraud_Data')
credit_results = train_and_evaluate(credit_X_train, credit_X_test, credit_y_train, credit_y_test, 'CreditCard')

# Save results for reporting
print("\n--- Saving Model Results for Reporting ---")
results_file_path = os.path.join(project_root, 'reports', 'model_results.txt')
with open(results_file_path, 'w') as f:
    f.write("Fraud Data Results:\n")
    for model, metrics in fraud_results.items():
        f.write(f"{model}:\n")
        f.write(f"  AUC-PR: {metrics['AUC-PR']:.4f}\n")
        f.write(f"  F1-Score: {metrics['F1-Score']:.4f}\n")
        f.write(f"  Confusion Matrix:\n{metrics['Confusion Matrix']}\n\n")
    
    f.write("Credit Card Data Results:\n")
    for model, metrics in credit_results.items():
        f.write(f"{model}:\n")
        f.write(f"  AUC-PR: {metrics['AUC-PR']:.4f}\n")
        f.write(f"  F1-Score: {metrics['F1-Score']:.4f}\n")
        f.write(f"  Confusion Matrix:\n{metrics['Confusion Matrix']}\n\n")
print(f"Model results saved to: {results_file_path}")

print("\nModel Training and Evaluation script completed successfully.")
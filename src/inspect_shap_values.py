# inspect_shap_values.py
import numpy as np
import os

# Assuming you run this script from the project root: C:\FraudDetection\fraud-detection-10academy> python inspect_shap_values.py
REPORTS_PATH = os.path.join('reports')

try:
    fraud_shap_values = np.load(os.path.join(REPORTS_PATH, 'shap_values_fraud.npy'))
    credit_shap_values = np.load(os.path.join(REPORTS_PATH, 'shap_values_creditcard.npy'))

    print("SHAP values loaded successfully!")
    print("\nFraud SHAP values shape:", fraud_shap_values.shape)
    print("Credit Card SHAP values shape:", credit_shap_values.shape)

    print("\nFirst 5 SHAP values for the first instance (Fraud Data):")
    print(fraud_shap_values[0, :5])

except FileNotFoundError as e:
    print(f"Error: .npy file not found. Make sure the path is correct and model_explainability.py ran successfully.")
    print(f"Missing file: {e}")
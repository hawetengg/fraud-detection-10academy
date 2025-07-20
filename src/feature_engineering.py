import pandas as pd # <--- Make sure this is present
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder # <--- Make sure this is present
from imblearn.over_sampling import SMOTE # <--- Make sure this is present (though you'll apply it later)

# Define base directory relative to this script's location
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
PROCESSED_DATA_PATH = os.path.join(project_root, 'data', 'processed')

# Load the previously engineered data
print("--- Loading engineered data for transformation ---")
# Ensure the correct file name is used: Fraud_Data_engineered.csv
# This is the output from the previous run of this script.
fraud_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'Fraud_Data_engineered.csv'))
print(f"Loaded Fraud Data: {fraud_data.shape[0]} rows, {fraud_data.shape[1]} columns")

# Ensure purchase_time and signup_time are datetime objects if needed for any future steps
# (They should be handled correctly by now, but good to ensure if not dropped)
if 'purchase_time' in fraud_data.columns and not pd.api.types.is_datetime64_any_dtype(fraud_data['purchase_time']):
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
if 'signup_time' in fraud_data.columns and not pd.api.types.is_datetime64_any_dtype(fraud_data['signup_time']):
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])


print("\n--- Encoding Categorical Features ---")
# Encode categorical features
categorical_cols = ['source', 'browser', 'sex', 'country']

# Check if categorical columns actually exist in the DataFrame before encoding
# This prevents errors if columns were dropped in previous steps or named differently.
existing_categorical_cols = [col for col in categorical_cols if col in fraud_data.columns]
if not existing_categorical_cols:
    print("Warning: No specified categorical columns found in the DataFrame to encode.")
else:
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    # Use .copy() to avoid SettingWithCopyWarning later if fraud_data is a view
    encoded_features = pd.DataFrame(encoder.fit_transform(fraud_data[existing_categorical_cols]),
                                    columns=encoder.get_feature_names_out(existing_categorical_cols))

    # Reset index of fraud_data before concat to ensure proper alignment
    fraud_data = fraud_data.reset_index(drop=True)
    encoded_features = encoded_features.reset_index(drop=True)

    # Concatenate encoded features and drop original categorical columns
    fraud_data = pd.concat([fraud_data.drop(existing_categorical_cols, axis=1), encoded_features], axis=1)
    print(f"Encoded {len(existing_categorical_cols)} categorical columns. New shape: {fraud_data.shape}")


print("\n--- Normalizing/Scaling Numerical Features ---")
# Normalize/Scale numerical features
# Ensure these columns exist before attempting to scale
numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'transaction_count', 'transaction_velocity']
existing_numerical_cols = [col for col in numerical_cols if col in fraud_data.columns]

if not existing_numerical_cols:
    print("Warning: No specified numerical columns found in the DataFrame to scale.")
else:
    scaler = StandardScaler()
    fraud_data[existing_numerical_cols] = scaler.fit_transform(fraud_data[existing_numerical_cols])
    print(f"Scaled {len(existing_numerical_cols)} numerical columns.")


# Handle class imbalance (apply SMOTE to training data only later, during model training)
# Note: SMOTE should be applied AFTER splitting into train/test to prevent data leakage.
# So, this part remains a comment or a reminder for a later stage.
print("\nNote: SMOTE (for class imbalance) will be applied during model training to avoid data leakage.")

# Save transformed dataset
print("\n--- Saving Transformed Dataset ---")
TRANSFORMED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'Fraud_Data_transformed.csv')
fraud_data.to_csv(TRANSFORMED_DATA_PATH, index=False)
print(f"Transformed dataset saved to: {TRANSFORMED_DATA_PATH}")

print("\nFeature Engineering and Transformation script completed successfully.")
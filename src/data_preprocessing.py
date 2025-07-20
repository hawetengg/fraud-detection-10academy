import pandas as pd
import numpy as np # Import numpy for searchsorted
import os

# Define base directory for data relative to this script's location
# This assumes the script is in 'src/' and data folders are siblings to 'src/'
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Construct full paths to data files
RAW_DATA_PATH = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(project_root, 'data', 'processed')

# Ensure processed data directory exists
# This will create 'fraud-detection-10academy/data/processed' if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
print(f"Ensured processed data directory exists at: {PROCESSED_DATA_PATH}")


# Load datasets
print("\n--- Loading Raw Datasets ---")
fraud_data = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Fraud_Data.csv'))
ip_to_country = pd.read_csv(os.path.join(RAW_DATA_PATH, 'IpAddress_to_Country.csv'))
creditcard_data = pd.read_csv(os.path.join(RAW_DATA_PATH, 'creditcard.csv'))

print(f"Loaded Fraud Data: {fraud_data.shape[0]} rows, {fraud_data.shape[1]} columns")
print(f"Loaded IP to Country Data: {ip_to_country.shape[0]} rows, {ip_to_country.shape[1]} columns")
print(f"Loaded Credit Card Data: {creditcard_data.shape[0]} rows, {creditcard_data.shape[1]} columns")

# --- Start Preprocessing ---

# 1. Check for and remove duplicates
print("\n--- Checking and Removing Duplicates ---")

initial_fraud_shape = fraud_data.shape
fraud_data = fraud_data.drop_duplicates()
print(f"Fraud Data: Original {initial_fraud_shape}, After duplicates: {fraud_data.shape}")

initial_creditcard_shape = creditcard_data.shape
creditcard_data = creditcard_data.drop_duplicates()
print(f"Credit Card Data: Original {initial_creditcard_shape}, After duplicates: {creditcard_data.shape}")

initial_ip_to_country_shape = ip_to_country.shape
ip_to_country = ip_to_country.drop_duplicates()
print(f"IP to Country Data: Original {initial_ip_to_country_shape}, After duplicates: {ip_to_country.shape}")


# 2. Check for missing values
print("\n--- Checking for Missing Values (after duplicate removal) ---")
print("Fraud Data Missing Values:\n", fraud_data.isnull().sum())
print("\nIP to Country Missing Values:\n", ip_to_country.isnull().sum())
print("\nCredit Card Data Missing Values:\n", creditcard_data.isnull().sum()) # Corrected


# 3. Handle missing values (if any)
print("\n--- Handling Missing Values (if any) ---")
initial_fraud_rows = fraud_data.shape[0]
initial_creditcard_rows = creditcard_data.shape[0]

fraud_data = fraud_data.dropna()
creditcard_data = creditcard_data.dropna()

if fraud_data.shape[0] < initial_fraud_rows:
    print(f"Dropped {initial_fraud_rows - fraud_data.shape[0]} rows from Fraud Data due to missing values.")
else:
    print("No rows dropped from Fraud Data (no missing values found).")

if creditcard_data.shape[0] < initial_creditcard_rows:
    print(f"Dropped {initial_creditcard_rows - creditcard_data.shape[0]} rows from Credit Card Data due to missing values.")
else:
    print("No rows dropped from Credit Card Data (no missing values found).")


# 4. Correct data types (Crucial for IP mapping)
print("\n--- Correcting Data Types ---")
# Ensure datetime columns are correctly parsed. errors='coerce' turns invalid dates into NaT
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], errors='coerce')
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], errors='coerce')
# Handle any NaNs created by invalid date parsing if necessary
fraud_data.dropna(subset=['signup_time', 'purchase_time'], inplace=True)

# Ensure ip_address is an integer type. Handle potential NaNs before converting to int if present
# (though your data seems clean here, it's good practice)
fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
print("Fraud Data dtypes after conversion:\n", fraud_data[['signup_time', 'purchase_time', 'ip_address']].dtypes)


ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(int)
ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(int)
print("\nIP to Country dtypes after conversion:\n", ip_to_country[['lower_bound_ip_address', 'upper_bound_ip_address']].dtypes)


# --- OPTIMIZED IP Address to Country Mapping ---
print("\n--- Mapping IP Addresses to Countries (Optimized) ---")

# Ensure ip_to_country is sorted by lower_bound_ip_address for searchsorted to work correctly
ip_to_country = ip_to_country.sort_values(by='lower_bound_ip_address').reset_index(drop=True)

# Use numpy.searchsorted for efficient lookup
# 'right' finds the index where `ip` would be inserted to maintain order,
# essentially pointing to the first element *greater than* ip.
idx = np.searchsorted(ip_to_country['lower_bound_ip_address'].values, fraud_data['ip_address'].values, side='right')

# Adjust indices to point to the potential *matching* range.
# We go back one step. If ip is smaller than the first lower_bound, idx will be 0, so idx-1 would be -1.
# np.maximum(idx - 1, 0) handles this to avoid negative indices.
idx_adj = np.maximum(idx - 1, 0)

# Initialize country column with a default 'Unknown'
fraud_data['country'] = 'Unknown'

# Filter for valid matches where the IP falls within the bounds of the identified row
# Only update if the ip_address is >= lower_bound AND <= upper_bound
# We use .values for faster comparison and direct numpy array manipulation
valid_indices = (fraud_data['ip_address'].values >= ip_to_country['lower_bound_ip_address'].iloc[idx_adj].values) & \
                (fraud_data['ip_address'].values <= ip_to_country['upper_bound_ip_address'].iloc[idx_adj].values)

# Apply the country based on valid matches
fraud_data.loc[valid_indices, 'country'] = ip_to_country['country'].iloc[idx_adj[valid_indices]].values

print("IP to country mapping complete.")
print(f"Fraud Data with new 'country' column: {fraud_data.shape[0]} rows, {fraud_data.shape[1]} columns")
print("Sample countries after mapping:\n", fraud_data['country'].value_counts().head())
print(f"Number of 'Unknown' countries: {(fraud_data['country'] == 'Unknown').sum()}")


# Save cleaned and type-corrected datasets
print("\n--- Saving Processed Data ---")
# It's good practice to save the `fraud_data` that was just modified (with country)
# as the main 'cleaned' version, and also with the specific 'with_country' name.
fraud_data.to_csv(os.path.join(PROCESSED_DATA_PATH, 'Fraud_Data_cleaned.csv'), index=False)
print("Saved: Fraud_Data_cleaned.csv")

creditcard_data.to_csv(os.path.join(PROCESSED_DATA_PATH, 'creditcard_cleaned.csv'), index=False)
print("Saved: creditcard_cleaned.csv")

ip_to_country.to_csv(os.path.join(PROCESSED_DATA_PATH, 'IpAddress_to_Country_cleaned.csv'), index=False)
print("Saved: IpAddress_to_Country_cleaned.csv")

fraud_data.to_csv(os.path.join(PROCESSED_DATA_PATH, 'Fraud_Data_with_country.csv'), index=False)
print("Saved: Fraud_Data_with_country.csv")


print(f"\nAll processed data saved to '{PROCESSED_DATA_PATH}' directory.")
print("Script execution complete.")
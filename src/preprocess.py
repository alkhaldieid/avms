import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks")

TRANSACTIONS_FILE = os.path.join(DATA_DIR, "snp_dld_2024_transactions.csv")
RENTS_FILE = os.path.join(DATA_DIR, "snp_dld_2024_rents.csv")
CLEANED_SALES_FILE = os.path.join(OUTPUT_DIR, "cleaned_sales_data.csv")
CLEANED_RENTALS_FILE = os.path.join(OUTPUT_DIR, "cleaned_rentals_data.csv")


def preprocess():
    # Load datasets
    sales_data = pd.read_csv(TRANSACTIONS_FILE, low_memory=False)
    rentals_data = pd.read_csv(RENTS_FILE, low_memory=False)

    # Step 1: Eliminate unnecessary columns
    sales_drop_columns = [
        'transaction_number', 'entry_id', 'meta_ts', 'master_project_en',
        'master_project_ar', 'property_type_ar', 'property_subtype_ar',
        'rooms_ar', 'project_name_ar', 'area_ar', 'nearest_landmark_ar',
        'nearest_metro_ar', 'nearest_mall_ar', 'parcel_id', 'transaction_type_en',
        'transaction_subtype_en', 'property_id', 'property_type_id', 
        'property_subtype_id', 'building_age', 'area_id', 'is_freehold_text'

    ]
    rentals_drop_columns = [
        'ejari_contract_number', 'land_property_id', 'entry_id', 'meta_ts',
        'master_project_en', 'master_project_ar', 'property_type_ar',
        'property_subtype_ar', 'property_usage_ar', 'project_name_ar',
        'area_ar', 'nearest_landmark_ar', 'nearest_metro_ar', 
        'nearest_mall_ar', 'parcel_id', 'property_id', 'property_usage_id',
        'area_id', 'ejari_property_type_id', 'ejari_property_sub_type_id'
    ]
    sales_data.drop(columns=sales_drop_columns, inplace=True, errors='ignore')
    rentals_data.drop(columns=rentals_drop_columns, inplace=True, errors='ignore')

    # Step 2: Handle date columns
    def process_date_columns(df, date_columns):
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
                df[f"{col}_dayofyear"] = df[col].dt.dayofyear
        df.drop(columns=date_columns, inplace=True, errors='ignore')
        return df

    sales_date_columns = ['transaction_datetime', 'req_from', 'req_to']
    rentals_date_columns = ['registration_date', 'contract_start_date', 'contract_end_date', 'req_from', 'req_to']

    sales_data = process_date_columns(sales_data, sales_date_columns)
    rentals_data = process_date_columns(rentals_data, rentals_date_columns)

    # Step 3: Drop constant columns
    def find_constant_columns(df):
        return [col for col in df.columns if df[col].nunique() == 1]

    sales_constant_columns = find_constant_columns(sales_data)
    rentals_constant_columns = find_constant_columns(rentals_data)

    sales_data.drop(columns=sales_constant_columns, inplace=True)
    rentals_data.drop(columns=rentals_constant_columns, inplace=True)

    # Step 4: Handle boolean columns
    def process_boolean_columns(df, boolean_columns):
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].map({'True': 1, 'False': 0, 'Yes': 1, 'No': 0, 'T': 1, 'F': 0}).fillna(0).astype(int)
        return df

    sales_boolean_columns = ['is_freehold', 'is_offplan']
    rentals_boolean_columns = ['is_freehold']

    sales_data = process_boolean_columns(sales_data, sales_boolean_columns)
    rentals_data = process_boolean_columns(rentals_data, rentals_boolean_columns)

    # Step 5: Encode categorical columns
    def process_categorical_columns(df, categorical_columns):
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    sales_categorical_columns = [
        'transaction_type_en', 'transaction_subtype_en', 'registration_type_en',
        'property_type_en', 'property_subtype_en', 'rooms_en', 'project_name_en', 'area_en'
    ]
    rentals_categorical_columns = [
        'property_type_en', 'property_subtype_en', 'property_usage_en', 'project_name_en', 'area_en'
    ]

    sales_data = process_categorical_columns(sales_data, sales_categorical_columns)
    rentals_data = process_categorical_columns(rentals_data, rentals_categorical_columns)

    # Step 6: Handle missing values
    sales_data['transaction_size_sqm'].fillna(sales_data['transaction_size_sqm'].median(), inplace=True)
    sales_data['property_usage_id'].fillna(sales_data['property_usage_id'].mode()[0], inplace=True)

    rentals_data['annual_amount'].fillna(rentals_data['annual_amount'].median(), inplace=True)

    # Step 7: Scale numerical columns
    def scale_numerical_columns(df, exclude_columns=[]):
        numerical_columns = df.select_dtypes(include=['number']).columns.difference(exclude_columns)
        scaler = MinMaxScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        return df

    exclude_sales = sales_boolean_columns + sales_categorical_columns
    exclude_rentals = rentals_boolean_columns + rentals_categorical_columns

    sales_data = scale_numerical_columns(sales_data, exclude_columns=exclude_sales)
    rentals_data = scale_numerical_columns(rentals_data, exclude_columns=exclude_rentals)

    # Step 8: Save cleaned data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sales_data.to_csv(CLEANED_SALES_FILE, index=False)
    rentals_data.to_csv(CLEANED_RENTALS_FILE, index=False)

    print(f"Data preprocessing complete.\nSales data saved to {CLEANED_SALES_FILE}\nRentals data saved to {CLEANED_RENTALS_FILE}")


if __name__ == "__main__":
    preprocess()

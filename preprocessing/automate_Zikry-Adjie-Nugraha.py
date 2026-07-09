"""Automasi preprocessing dataset Telco Customer Churn.

Konversi dari proses eksperimen pada Eksperimen_Zikry-Adjie-Nugraha.ipynb --
tahapannya sama persis (drop identifier, fix TotalCharges, handle missing value,
normalisasi kategori, encoding), tapi dibungkus jadi fungsi yang bisa dipanggil
otomatis (dipakai ulang di Kriteria 3 - Workflow CI).
"""

import argparse
import os

import pandas as pd


def load_raw_dataset(raw_path):
    #Load the Telco Customer Churn dataset
    df = pd.read_csv(raw_path)
    return df


def clean_missing_values(df):
    #Dropping identifier column (not a predictive feature)
    df_clean = df.drop(columns=['customerID'])

    #Fixing TotalCharges data type (object -> numeric)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    #Removing data with NaN value (hidden missing value pada TotalCharges)
    prev_size = df_clean.shape[0]
    df_clean.dropna(how='any', inplace=True)
    current_size = df_clean.shape[0]
    print('Dari data cleaning, kita membuang {}% data karena NaN value'.format(
        round(((prev_size - current_size) / prev_size) * 100, 2)
    ))

    return df_clean


def normalize_categories(df_clean):
    #Normalizing "No internet service" / "No phone service" into "No" (semantically the same)
    service_cols_with_no_internet = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                      'TechSupport', 'StreamingTV', 'StreamingMovies']

    for col in service_cols_with_no_internet:
        df_clean[col] = df_clean[col].replace('No internet service', 'No')

    df_clean['MultipleLines'] = df_clean['MultipleLines'].replace('No phone service', 'No')

    return df_clean


def encode_features(df_clean):
    #Label and one hot encoding process
    #Handling column with 2 distinct value (binary Yes/No & gender)
    binary_yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

    for col in binary_yes_no_cols:
        df_clean[col] = df_clean[col].map({'No': 0, 'Yes': 1}).astype(int)

    df_clean['gender'] = df_clean['gender'].map({'Female': 0, 'Male': 1}).astype(int)

    #Handling column with more than 2 distinct value (one hot encoding)
    multi_category_cols = ['Contract', 'InternetService', 'PaymentMethod']
    df_clean = pd.get_dummies(df_clean, columns=multi_category_cols, drop_first=False)

    return df_clean


def preprocess_data(raw_path, output_path=None):
    """Fungsi utama: preprocessing otomatis dari raw CSV -> data siap dilatih.

    Args:
        raw_path: path ke file Telco-Customer-Churn.csv (raw)
        output_path: kalau diisi, hasil preprocessing disimpan ke path ini (CSV)

    Returns:
        pandas.DataFrame hasil preprocessing (siap untuk split X/y & train_test_split)
    """
    df = load_raw_dataset(raw_path)
    df_clean = clean_missing_values(df)
    df_clean = normalize_categories(df_clean)
    df_clean = encode_features(df_clean)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f'Preprocessed dataset disimpan di: {output_path}')

    return df_clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automate Telco Customer Churn preprocessing')
    parser.add_argument(
        '--raw_path',
        default='../namadataset_raw/Telco-Customer-Churn.csv',
        help='Path ke file dataset raw (default: ../namadataset_raw/Telco-Customer-Churn.csv)',
    )
    parser.add_argument(
        '--output_path',
        default='namadataset_preprocessing/telco_churn_preprocessed.csv',
        help='Path output CSV hasil preprocessing',
    )
    args = parser.parse_args()

    preprocess_data(args.raw_path, args.output_path)

# trigger CI run
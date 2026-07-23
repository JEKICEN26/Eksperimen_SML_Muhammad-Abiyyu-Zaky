"""
Automate Preprocessing - Muhammad Abiyyu Zaky
================================================
Script otomatis untuk melakukan preprocessing dataset California Housing.
Mengembalikan data yang siap dilatih (X_train, X_test, y_train, y_test).

Tahapan preprocessing:
1. Load dataset (dari sklearn atau CSV)
2. Hapus duplikat
3. Handle missing values
4. Deteksi & cap outlier (IQR)
5. Feature engineering
6. Train-test split
7. Standarisasi fitur (StandardScaler)
8. Simpan hasil preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(source='sklearn', filepath=None):
    """
    Load dataset California Housing.
    
    Parameters:
    -----------
    source : str
        Sumber data ('sklearn' atau 'csv')
    filepath : str, optional
        Path ke file CSV jika source='csv'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame berisi dataset mentah
    """
    if source == 'csv' and filepath is not None:
        df = pd.read_csv(filepath)
        print(f"[LOAD] Dataset dimuat dari CSV: {filepath}")
    else:
        california = fetch_california_housing()
        df = pd.DataFrame(california.data, columns=california.feature_names)
        df['MedHouseVal'] = california.target
        print("[LOAD] Dataset dimuat dari sklearn")
    
    print(f"[LOAD] Shape: {df.shape}")
    return df


def remove_duplicates(df):
    """
    Hapus baris duplikat dari DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
    
    Returns:
    --------
    pd.DataFrame
        DataFrame tanpa duplikat
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"[DUPLIKAT] Dihapus: {removed} baris (sisa: {after})")
    return df


def handle_missing_values(df):
    """
    Handle missing values dengan mengisi menggunakan median.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
    
    Returns:
    --------
    pd.DataFrame
        DataFrame tanpa missing values
    """
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"[MISSING] {col}: diisi dengan median ({median_val:.4f})")
        print(f"[MISSING] Total missing values ditangani: {missing_total}")
    else:
        print("[MISSING] Tidak ada missing values")
    return df


def cap_outliers_iqr(df, columns):
    """
    Cap outlier menggunakan metode IQR (Interquartile Range).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
    columns : list
        Daftar kolom yang akan di-cap outliernya
    
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan outlier yang sudah di-cap
    """
    df_capped = df.copy()
    for col in columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = len(df_capped[(df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)])
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"[OUTLIER] {col}: {outliers_count} outlier di-cap "
              f"(range: [{lower_bound:.2f}, {upper_bound:.2f}])")
    
    return df_capped


def feature_engineering(df):
    """
    Buat fitur baru dari fitur yang ada.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input
    
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan fitur tambahan
    """
    # Rasio kamar tidur terhadap total kamar
    df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
    
    # Rasio kamar per orang
    df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup']
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    print("[FEATURE] Fitur baru ditambahkan: BedroomRatio, RoomsPerPerson")
    print(f"[FEATURE] Total fitur: {df.shape[1]}")
    
    return df


def split_and_scale(df, target_col='MedHouseVal', test_size=0.2, random_state=42):
    """
    Split data dan lakukan standarisasi.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang sudah dipreprocessing
    target_col : str
        Nama kolom target
    test_size : float
        Proporsi data test
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns)
    """
    # Pisahkan fitur dan target
    feature_columns = [col for col in df.columns if col != target_col]
    X = df[feature_columns]
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standarisasi
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_columns,
        index=X_test.index
    )
    
    print(f"[SPLIT] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"[SCALE] StandardScaler diterapkan")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns


def save_preprocessed_data(df, output_path):
    """
    Simpan dataset yang sudah dipreprocessing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame hasil preprocessing
    output_path : str
        Path output file CSV
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SAVE] Dataset disimpan ke: {output_path}")
    print(f"[SAVE] Shape: {df.shape}")


def run_preprocessing(source='sklearn', filepath=None, output_path=None):
    """
    Jalankan seluruh pipeline preprocessing.
    
    Parameters:
    -----------
    source : str
        Sumber data ('sklearn' atau 'csv')
    filepath : str, optional
        Path ke file CSV
    output_path : str, optional
        Path untuk menyimpan hasil preprocessing
    
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE - California Housing Dataset")
    print("=" * 60)
    
    # 1. Load data
    print("\n--- Tahap 1: Load Data ---")
    df = load_data(source=source, filepath=filepath)
    
    # 2. Hapus duplikat
    print("\n--- Tahap 2: Hapus Duplikat ---")
    df = remove_duplicates(df)
    
    # 3. Handle missing values
    print("\n--- Tahap 3: Handle Missing Values ---")
    df = handle_missing_values(df)
    
    # 4. Cap outlier
    print("\n--- Tahap 4: Cap Outlier (IQR) ---")
    cols_to_cap = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    df = cap_outliers_iqr(df, cols_to_cap)
    
    # 5. Feature engineering
    print("\n--- Tahap 5: Feature Engineering ---")
    df = feature_engineering(df)
    
    # 6. Simpan dataset preprocessed
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'california_housing_preprocessing.csv'
        )
    print(f"\n--- Tahap 6: Simpan Dataset ---")
    save_preprocessed_data(df, output_path)
    
    # 7. Split dan scale
    print("\n--- Tahap 7: Split & Scale ---")
    X_train, X_test, y_train, y_test, scaler, feature_columns = split_and_scale(df)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print(f"Total fitur: {len(feature_columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns


# Main execution
if __name__ == "__main__":
    # Cek apakah raw CSV ada
    raw_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'california_housing_raw.csv'
    )
    
    if os.path.exists(raw_csv_path):
        X_train, X_test, y_train, y_test, scaler, features = run_preprocessing(
            source='csv', filepath=raw_csv_path
        )
    else:
        X_train, X_test, y_train, y_test, scaler, features = run_preprocessing(
            source='sklearn'
        )
    
    # Tampilkan ringkasan
    print("\n=== RINGKASAN DATA SIAP LATIH ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Fitur: {features}")

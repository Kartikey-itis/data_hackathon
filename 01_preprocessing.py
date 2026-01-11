import pandas as pd
import glob
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# BASE PATH (UPDATE YOURS)
BASE_PATH = r"A:\Code\Hackathons\Data hackathon 2026"

# Dataset folders
DATASETS = {
    'enrolment': 'api_data_aadhar_enrolment',
    'demographic': 'api_data_aadhar_demographic',
    'biometric': 'api_data_aadhar_biometric'
}


def load_dataset_chunk(folder, pattern):
    """Load all CSV chunks from a folder"""
    files = glob.glob(os.path.join(BASE_PATH, folder, pattern))
    print(f"Found {len(files)} files in {folder}")

    dfs = []
    for file in tqdm(files, desc=f"Loading {folder}"):
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {combined.shape}")
    return combined


def explore_columns(df, dataset_name):
    """Quick EDA"""
    print(f"\n=== {dataset_name} Dataset ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nMemory usage:", df.memory_usage(deep=True).sum() / 1024 ** 2, "MB")
    print("\nSample rows:")
    print(df.head(2))
    print("\nNull counts:")
    print(df.isnull().sum().head(10))
    return df.columns.tolist()


# MAIN PIPELINE
if __name__ == "__main__":
    print("🚀 Starting UIDAI Data Preprocessing...")

    # Step 1: Load all datasets
    enrolment_df = load_dataset_chunk(DATASETS['enrolment'], "*.csv")
    demographic_df = load_dataset_chunk(DATASETS['demographic'], "*.csv")
    biometric_df = load_dataset_chunk(DATASETS['biometric'], "*.csv")

    # Step 2: Quick EDA
    explore_columns(enrolment_df, "Enrolment")
    explore_columns(demographic_df, "Demographic")
    explore_columns(biometric_df, "Biometric")

    # Save raw combined data
    enrolment_df.to_parquet('data/raw/enrolment_raw.parquet', index=False)
    demographic_df.to_parquet('data/raw/demographic_raw.parquet', index=False)
    biometric_df.to_parquet('data/raw/biometric_raw.parquet', index=False)

    print("✅ Raw data saved!")

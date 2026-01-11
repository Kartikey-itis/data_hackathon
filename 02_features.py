import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

BASE_PATH = r"A:\Code\Hackathons\Data hackathon 2026"


def load_full_dataset(folder_name):
    all_files = glob.glob(os.path.join(BASE_PATH, folder_name, "*", "*.csv"))
    chunks = []
    for file in all_files:
        chunk = pd.read_csv(file, low_memory=False)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


print("🚀 Loading FULL datasets...")
enrolment = load_full_dataset("api_data_aadhar_enrolment")
demographic = load_full_dataset("api_data_aadhar_demographic")
biometric = load_full_dataset("api_data_aadhar_biometric")

print("Shapes:", enrolment.shape, demographic.shape, biometric.shape)


# 🔧 FEATURE ENGINEERING
def engineer_features(df, dataset_type):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Dataset-specific total activity
    if dataset_type == 'enrolment':
        df['total_activity'] = df[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
    elif dataset_type == 'demographic':
        df['total_activity'] = df[['demo_age_5_17', 'demo_age_17_']].sum(axis=1)
    else:  # biometric
        df['total_activity'] = df[['bio_age_5_17', 'bio_age_17_']].sum(axis=1)

    return df


enrolment = engineer_features(enrolment, 'enrolment')
demographic = engineer_features(demographic, 'demographic')
biometric = engineer_features(biometric, 'biometric')


# 💎 CREATE DISTRICT-LEVEL FEATURES (NO CENTRE_ID, so district+pincode)
def create_district_features(df):
    """Daily district+pincode level aggregation"""
    agg_dict = {
        'total_activity': ['sum', 'mean', 'std'],
        'year': 'first',
        'month': 'first',
        'day_of_week': 'first'
    }

    # Age-specific features
    age_cols = [col for col in df.columns if col.startswith(('age_', 'demo_', 'bio_'))]
    for col in age_cols:
        agg_dict[col] = ['sum', 'mean']

    district_features = df.groupby(['state', 'district', 'pincode']).agg(agg_dict).reset_index()

    # Flatten columns
    new_cols = []
    for col in district_features.columns:
        if isinstance(col, tuple):
            new_cols.append('_'.join([str(c) for c in col if c != '']))
        else:
            new_cols.append(col)
    district_features.columns = new_cols

    return district_features


# Generate features for each dataset
print("Creating district-level features...")
enrol_dist = create_district_features(enrolment)
demo_dist = create_district_features(demographic)
bio_dist = create_district_features(biometric)

print("Shapes after aggregation:", enrol_dist.shape, demo_dist.shape, bio_dist.shape)

# 🛠️ COMBINE (safer merge - check columns first)
print("Merging datasets...")
print("Enrolment columns:", enrol_dist[['state', 'district', 'pincode']].columns.tolist())
print("Demo columns:", demo_dist[['state', 'district', 'pincode']].columns.tolist())

# Merge step by step
master = enrol_dist.merge(demo_dist, on=['state', 'district', 'pincode'], how='outer', suffixes=('_enrol', '_demo'))
master = master.merge(bio_dist, on=['state', 'district', 'pincode'], how='outer', suffixes=('', '_bio'))

# Fill NaNs
numeric_cols = master.select_dtypes(include=[np.number]).columns
master[numeric_cols] = master[numeric_cols].fillna(0)

# CREATE ANOMALY LABELS
volume_cols = [col for col in master.columns if 'total_activity_sum' in col]
master['total_volume'] = master[volume_cols].sum(axis=1)
master['volume_spike'] = master['total_volume'] / master['total_volume'].rolling(30, min_periods=1).mean()
master['is_anomaly'] = (
        (master['volume_spike'] > 2.5) |
        (master['total_volume'] > master['total_volume'].quantile(0.95))
).astype(int)

print(f"✅ MASTER TABLE: {master.shape}")
print(f"Anomalies detected: {master['is_anomaly'].sum()}")

# SAVE
Path("data/processed").mkdir(exist_ok=True)
master.to_parquet('data/processed/master_anomaly_table.parquet', index=False)
print("🎉 XGBoost-READY data saved!")

print("\nTop 5 anomalies:")
print(
    master.nlargest(5, 'volume_spike')[['state', 'district', 'pincode', 'total_volume', 'volume_spike', 'is_anomaly']])

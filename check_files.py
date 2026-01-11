import os
print("Files in data/processed/:")
print(os.listdir('data/processed/'))
print("\nFull path check:")
print(os.path.exists('data/processed/master_anomaly_table.parquet'))

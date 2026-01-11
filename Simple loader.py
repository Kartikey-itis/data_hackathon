import pandas as pd
import glob
import os

# YOUR EXACT PATH
BASE_PATH = r"A:\Code\Hackathons\Data hackathon 2026"

# Load ONE file from each dataset to see columns
print("🔍 Checking column structure...")

enrol_file = glob.glob(os.path.join(BASE_PATH, "api_data_aadhar_enrolment", "*", "*.csv"))[0]
demo_file = glob.glob(os.path.join(BASE_PATH, "api_data_aadhar_demographic", "*", "*.csv"))[0]
bio_file = glob.glob(os.path.join(BASE_PATH, "api_data_aadhar_biometric", "*", "*.csv"))[0]

print(f"Loading sample: {enrol_file}")
enrolment_sample = pd.read_csv(enrol_file, nrows=1000)
print("ENROLMENT COLUMNS:", enrolment_sample.columns.tolist())
print("\nENROLMENT SAMPLE:")
print(enrolment_sample.head(2))

print(f"\nLoading sample: {demo_file}")
demographic_sample = pd.read_csv(demo_file, nrows=1000)
print("DEMOGRAPHIC COLUMNS:", demographic_sample.columns.tolist())
print("\nDEMOGRAPHIC SAMPLE:")
print(demographic_sample.head(2))

print(f"\nLoading sample: {bio_file}")
biometric_sample = pd.read_csv(bio_file, nrows=1000)
print("BIOMETRIC COLUMNS:", biometric_sample.columns.tolist())
print("\nBIOMETRIC SAMPLE:")
print(biometric_sample.head(2))

# Save samples for safety
enrolment_sample.to_csv('sample_enrolment.csv', index=False)
demographic_sample.to_csv('sample_demographic.csv', index=False)
biometric_sample.to_csv('sample_biometric.csv', index=False)

print("\n✅ SAMPLES SAVED! Share screenshot of column names above.")

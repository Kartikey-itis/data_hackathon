import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

print("🚀 Training XGBoost Anomaly Detector...")

# Load your parquet file
df = pd.read_parquet('data/processed/master_anomaly_table.parquet')
print(f"✅ Loaded: {df.shape}")
print(f"📊 Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.1%})")

# Features (exclude IDs)
feature_cols = [col for col in df.columns if col not in ['state', 'district', 'pincode', 'is_anomaly']]
X = df[feature_cols].fillna(0)
y = df['is_anomaly']

print(f"Training on {len(feature_cols)} features...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=6,  # Handle imbalance
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Results
y_pred = model.predict(X_test)
print("\n🎯 XGBoost Results:")
print(classification_report(y_test, y_pred))

# Add predictions to dataframe
df['xgb_anomaly_prob'] = model.predict_proba(X)[:,1]

# Save top anomalies
top_anomalies = df.nlargest(20, 'xgb_anomaly_prob')[['state', 'district', 'pincode', 'volume_spike', 'xgb_anomaly_prob', 'is_anomaly']]
top_anomalies.to_csv('top_anomalies.csv', index=False)

# Save model
os.makedirs('data/models', exist_ok=True)
model.save_model('data/models/anomaly_detector.json')

# Plot feature importance
xgb.plot_importance(model, max_num_features=10)
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ XGBoost COMPLETE!")
print("\n🔥 TOP 5 ANOMALIES:")
print(top_anomalies.head())

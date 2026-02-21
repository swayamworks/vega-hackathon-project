import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

np.random.seed(42)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("synthetic_nhai_accident_data.csv")
print("Dataset shape:", df.shape)

# Drop rows with missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude']).copy()

# Fill missing values
df['Alcohol_Involved'] = df['Alcohol_Involved'].fillna(0)
df['Hour'] = df['Hour'].fillna(df['Hour'].median())
df['Speed_Limit'] = df['Speed_Limit'].fillna(df['Speed_Limit'].median())
df['Weather'] = df['Weather'].fillna('Clear')

# Force correct data types
df['Alcohol_Involved'] = df['Alcohol_Involved'].astype(int)

# --------------------------------------------------
# 2. SPATIAL CLUSTERING
# --------------------------------------------------

coords = df[['Latitude', 'Longitude']]

dbscan = DBSCAN(eps=0.06, min_samples=12)
df['cluster_id'] = dbscan.fit_predict(coords)

print("\nCluster distribution:")
print(df['cluster_id'].value_counts())

# Cluster density feature
cluster_counts = df['cluster_id'].value_counts()
df['Cluster_Density'] = df['cluster_id'].map(cluster_counts)

# --------------------------------------------------
# 3. STRONG NON-LINEAR TARGET GENERATION
# --------------------------------------------------

df['Is_Rain'] = (df['Weather'] == 'Rain').astype(int)
df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 4)).astype(int)
df['High_Speed'] = (df['Speed_Limit'] >= 80).astype(int)

prob = np.full(len(df), 0.05)

# Strong nonlinear interactions (multiplication, not bitwise)
prob += 0.55 * (df['Is_Rain'] * df['Alcohol_Involved'])
prob += 0.40 * (df['Alcohol_Involved'] * df['High_Speed'])
prob += 0.35 * (df['Is_Night'] * df['High_Speed'])
prob += 0.25 * df['Alcohol_Involved']
prob += 0.20 * df['Is_Rain']
prob += 0.15 * df['Is_Night']

high_density = (df['Cluster_Density'] > df['Cluster_Density'].median()).astype(int)
prob += 0.20 * high_density

prob += np.random.normal(0, 0.02, len(df))
prob = np.clip(prob, 0.01, 0.95)

df['High_Risk'] = np.random.binomial(1, prob)

print("\nTarget distribution:")
print(df['High_Risk'].value_counts())

# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

df['Rain_Alcohol'] = df['Is_Rain'] * df['Alcohol_Involved']
df['Night_Alcohol'] = df['Is_Night'] * df['Alcohol_Involved']
df['Rain_HighSpeed'] = df['Is_Rain'] * df['High_Speed']

# --------------------------------------------------
# 5. PREPARE FEATURES
# --------------------------------------------------

X = df.drop(columns=[
    'Severity',
    'Fatality',
    'timestamp',
    'High_Risk'
], errors='ignore')

y = df['High_Risk']

X = pd.get_dummies(X, drop_first=True)

# --------------------------------------------------
# 6. GROUP-BASED SPLIT
# --------------------------------------------------

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=df['cluster_id'])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

model = HistGradientBoostingClassifier(
    max_depth=10,
    learning_rate=0.05,
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

y_probs = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)
print("\nROC-AUC:", roc_auc)

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("Best threshold:", best_threshold)

y_pred = (y_probs >= best_threshold).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

# --------------------------------------------------
# 9. FEATURE IMPORTANCE (PERMUTATION BASED)
# --------------------------------------------------

from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# --------------------------------------------------
# 10. VISUALIZATION
# --------------------------------------------------

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# --------------------------------------------------
# 11. EXPORT DASHBOARD DATA
# --------------------------------------------------

full_probs = model.predict_proba(X)[:, 1]

df['risk_score'] = full_probs

df['risk_level'] = pd.cut(
    df['risk_score'],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low", "Moderate", "Critical"]
)

dashboard_df = df[
    ['Latitude', 'Longitude', 'cluster_id',
     'risk_score', 'risk_level', 'timestamp']
].copy()

dashboard_df = dashboard_df.rename(columns={
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

dashboard_df.to_csv(
    "final_dashboard_dataset_hackathon.csv",
    index=False
)

print("\nHackathon dashboard dataset exported successfully.")
dashboard_df = df[
    ['Latitude', 'Longitude', 'cluster_id',
     'risk_score', 'risk_level', 'timestamp']
].copy()

dashboard_df = dashboard_df.rename(columns={
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

dashboard_df.to_csv(
    "final_dashboard_dataset_hackathon.csv",
    index=False
)

print("Model output saved successfully.")
# ==========================================
# TASK 4: DISEASE PREDICTION FROM MEDICAL DATA
# ==========================================
!pip install -q numpy pandas scikit-learn xgboost matplotlib seaborn openml

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

sns.set_theme(style='whitegrid')

# 1. Fetch Real Medical Dataset (Diabetes)
print("Fetching clinical dataset from OpenML...")
diabetes = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
df = diabetes.frame

# 2. Preprocess Data
X = df.drop(columns=['class'])
y = LabelEncoder().fit_transform(df['class']) # 0 = tested_negative, 1 = tested_positive

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train XGBoost Model
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluate & Visualize
y_pred = model.predict(X_test_scaled)

print("\n--- TASK 4 EVALUATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Medical Disease Prediction Matrix')
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.show()

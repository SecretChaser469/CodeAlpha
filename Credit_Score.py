# ==========================================
# TASK 1: CREDIT SCORING MODEL
# ==========================================
!pip install -q numpy pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

sns.set_theme(style='darkgrid')

# 1. Generate Synthetic Financial Data
X, y = make_classification(n_samples=2500, n_features=10, n_informative=7, random_state=42)
cols = ['income', 'debt_ratio', 'payment_history', 'age', 'employment_length',
        'savings', 'checking', 'credit_lines', 'open_accounts', 'utilization']
df = pd.DataFrame(X, columns=cols)
df['credit_risk'] = y

# 2. Preprocess Data
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['credit_risk']), df['credit_risk'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Model
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluate & Visualize
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("--- TASK 1 EVALUATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='crimson', lw=2, label=f'AUC: {roc_auc_score(y_test, y_prob):.3f}')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.title('Credit Risk Model ROC')
plt.legend()

plt.subplot(1, 2, 2)
sns.barplot(x=model.feature_importances_, y=cols, palette='viridis')
plt.title('Feature Importances in Credit Scoring')
plt.tight_layout()
plt.show()

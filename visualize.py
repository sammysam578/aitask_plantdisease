
# visualize_results.py
# Metrics + Visualizations for Random Forest Plant Health Classifier

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


# 1. Load trained model, scaler, and test data

rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test, y_test = joblib.load('test_data.pkl')

X_test_df = pd.DataFrame(X_test, columns=scaler.feature_names_in_)


# 2. Predict on test set

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]


# 3. Compute Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Healthy', 'Diseased'])

print("\n--- Overall Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\n--- Classification Report ---")
print(class_report)

print("\n--- Confusion Matrix ---")
print(conf_matrix)


# 4. Feature Importance

feat_importances = pd.Series(rf.feature_importances_, index=scaler.feature_names_in_)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# 5. Confusion Matrix Heatmap

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()


# 6. ROC Curve

y_bin = label_binarize(y_test, classes=[0,1])
fpr, tpr, _ = roc_curve(y_bin, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Plant Health Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# 7. Extra Visualizations
# -----------------------------
# 7a. Class Distribution (Test Set)
plt.figure(figsize=(6,4))
plt.pie(np.bincount(y_test), labels=['Healthy', 'Diseased'], autopct='%1.1f%%', colors=['skyblue','salmon'])
plt.title('Class Distribution in Test Set')
plt.tight_layout()
plt.show()

# 7b. Prediction vs Actual Counts
actual_counts = np.bincount(y_test)
pred_counts = np.bincount(y_pred)
labels = ['Healthy','Diseased']
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, actual_counts, width, label='Actual', color='skyblue')
plt.bar(x + width/2, pred_counts, width, label='Predicted', color='salmon')
plt.xticks(x, labels)
plt.ylabel('Count')
plt.title('Actual vs Predicted Class Counts')
plt.legend()
plt.tight_layout()
plt.show()

# 7c. Feature Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(X_test_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
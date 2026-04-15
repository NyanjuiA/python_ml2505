# Python file to demonstrate a Confusion matrix in evaluating predictions using
# a synthetic dataset

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Generate a realistic dataset
# Simulate  a binary classification problem with imbalance
X,y = make_classification(
   n_samples=1000,
   n_features=10,
   n_informative=5,
   n_redundant=2,
   n_classes=2,
   weights=[.7,.3],
   random_state=42,
)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
   X,y,
   test_size=0.3,
   stratify=y,           # Preserve class distribution
   random_state=42
)

# 3. Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predictions
# Default predictions (threshold = 0.5)
y_pred = model.predict(X_test)

# Probabilities for threshold tuning
y_probs = model.predict_proba(X_test)[:,1]

# 5. Confusion Matrix
conf_matrix = confusion_matrix(y_test,y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# 6. Classification Report
print(f"\nClassification Report:\n{classification_report(y_test,y_pred)}")

# 7. ROC-AUC Score
roc_score = roc_auc_score(y_test,y_probs)
print(f"\nROC AUC score:\n{roc_score:.3f}")

# 8. Visualisation
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Default Threshold = .5')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 9. Threshold Tuning
# Adjust the threshold to show impact on confusion matrix
threshold = .3
y_pred_custom = (y_probs >= threshold).astype(int)
conf_matrix_custom = confusion_matrix(y_test,y_pred_custom)

print(f"\nConfusion Matrix (Threshold = {threshold:.2f}):\n{conf_matrix_custom}")
print(f"\nClassification Report (Custom Threshold)\n{classification_report(y_test,y_pred_custom)}")

# Visualise the new/custom confusion matrix
disp_custom = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_custom)
disp_custom.plot(cmap='Oranges')
plt.title(f'Confusion Matrix (Custom Threshold = {threshold})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
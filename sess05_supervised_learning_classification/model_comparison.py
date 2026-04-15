# Python script to compare Logistic Regression and Decision Tree models based on
# their performance using accuracy and F1 score

"""
Improved Fruit Classification Example

This script demonstrates:
- Creating a realistic dataset with meaningful patterns
- Proper train/test splitting with stratification
- Feature scaling for Logistic Regression
- Regularization to prevent overfitting
- Model comparison using multiple metrics
"""

# ==============================
# Imports
# ==============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==============================
# 1. Reproducibility
# ==============================
np.random.seed(42)

# ==============================
# 2. Create a realistic dataset
# ==============================
num_samples_per_class = 200

# Apple (0): medium weight, medium colour
apple_weight = np.random.normal(loc=150, scale=10, size=num_samples_per_class)
apple_color = np.random.normal(loc=5, scale=1, size=num_samples_per_class)
apple_label = np.zeros(num_samples_per_class)

# Orange (1): heavier, higher colour intensity
orange_weight = np.random.normal(loc=180, scale=12, size=num_samples_per_class)
orange_color = np.random.normal(loc=7, scale=1, size=num_samples_per_class)
orange_label = np.ones(num_samples_per_class)

# Banana (2): lighter, lower colour intensity
banana_weight = np.random.normal(loc=120, scale=8, size=num_samples_per_class)
banana_color = np.random.normal(loc=3, scale=1, size=num_samples_per_class)
banana_label = np.full(num_samples_per_class, 2)

# Combine dataset
weights = np.concatenate([apple_weight, orange_weight, banana_weight])
colors = np.concatenate([apple_color, orange_color, banana_color])
labels = np.concatenate([apple_label, orange_label, banana_label])

# Create DataFrame
fruit_data = pd.DataFrame({
    'Weight': weights,
    'Colour_Intensity': colors,
    'Fruit_Type': labels
})

# ==============================
# 3. Features and target
# ==============================
X = fruit_data[['Weight', 'Colour_Intensity']]
y = fruit_data['Fruit_Type']

# ==============================
# 4. Train-test split (STRATIFIED)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # ensures balanced class distribution
    random_state=42
)

# ==============================
# 5. Logistic Regression Pipeline
# ==============================
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # important for LR
    ('model', LogisticRegression(max_iter=300))
])

log_reg_pipeline.fit(X_train, y_train)
y_pred_log = log_reg_pipeline.predict(X_test)

# ==============================
# 6. Decision Tree (Regularized)
# ==============================
decision_tree = DecisionTreeClassifier(
    max_depth=5,          # prevent overfitting
    min_samples_split=10,
    random_state=42
)

decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

# ==============================
# 7. Evaluation Function
# ==============================
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{name} Performance")
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


# ==============================
# 8. Results
# ==============================
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Decision Tree", y_test, y_pred_tree)











# # Import the required modules
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, f1_score,classification_report
# from sklearn.model_selection import train_test_split
#
# # Seed for reproducibility
# np.random.seed(42)
#
# # Generate a dummy fruit dataset
# num_samples = 200
#
# # Features: weight (grams) and colour intensity (scale 1 - 10)
# weights = np.random.normal(size=num_samples,loc=150,scale=30) # Avg. weight around 150 grams
# colour_intensity = np.random.uniform(size=num_samples,low=1,high=10) # colour intensity on a scale of 1 - 10
#
# # Item labels: 0 -> Apple, 1 -> Orange, 2 -> Banana
# labels = np.random.choice([0,1,2],size = num_samples)
#
# # Create a dataframe for easier handling
# fruit_data = pd.DataFrame({
#    'Weight': weights,
#    'Colour_Intensity': colour_intensity,
#    'Fruit_Type': labels
# })
#
# # Split features and target variable
# X = fruit_data[['Weight', 'Colour_Intensity']]
# y = fruit_data['Fruit_Type']
#
# # Split the data to training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 1. Logistic Regression Model
# log_reg = LogisticRegression(max_iter=200) # increase iterations to ensure convergence
# log_reg.fit(X_train, y_train)
# y_pred_log_reg = log_reg.predict(X_test)
# accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
# f1_log_reg = f1_score(y_test, y_pred_log_reg,average='weighted')
#
# # 2. Decision Tree Model
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, y_train)
# y_pred_tree = decision_tree.predict(X_test)
# accuracy_tree = accuracy_score(y_test, y_pred_tree)
# f1_tree = f1_score(y_test, y_pred_tree,average='weighted')
#
# # Display results
# print(f"Logistic Regression Accuracy: {accuracy_log_reg:.4f}, "
#       f"Logistic Regression F1 Score: {f1_log_reg:.4f}")
# print(f"Decision Tree Accuracy: {accuracy_tree}, Decision Tree F1 Score: {f1_tree:.4f}")
# print(f"\nClassification Report for Logistic Regression:"
#       f"\n{classification_report(y_test, y_pred_log_reg)} ")
# print(f"\nClassification Report for Decision Tree Classifier:"
#       f"\n{classification_report(y_test, y_pred_tree)} ")
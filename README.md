# STEP 0: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

# STEP 1: LOAD AND CLEAN DATA
# Make sure to upload your CSV file as 'dataset.csv' using the left panel in Colab
df = pd.read_csv('dataset.csv')

# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Encode 'diagnosis': M = 1 (Malignant), B = 0 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# STEP 2: SPLIT AND STANDARDIZE
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 3: TRAIN LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# STEP 4: EVALUATE MODEL
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("🔍 Precision:", precision)
print("🔍 Recall:", recall)
print("🔍 ROC-AUC Score:", roc_auc)

# PLOT CONFUSION MATRIX & ROC CURVE
plt.figure(figsize=(15, 6))

# Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()

# STEP 5: SIGMOID FUNCTION EXPLANATION
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot Sigmoid Function
z_vals = np.linspace(-10, 10, 100)
sigmoid_vals = sigmoid(z_vals)

plt.figure(figsize=(6, 4))
plt.plot(z_vals, sigmoid_vals, color='green')
plt.title("Sigmoid Function")
plt.xlabel("z (logit)")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

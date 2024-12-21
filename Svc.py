import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv(r"C:\Users\MSI\Downloads\ParisHousingClass.csv")

# Feature and target selection (ensure y is 1D)
X = dataset.iloc[:, 2:3].values  # Select feature columns
y = dataset.iloc[:, -1].values   # Select target (flatten to 1D)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling (use the same scaler for both training and testing)
scale1 = StandardScaler()
X_train = scale1.fit_transform(X_train)
X_test = scale1.transform(X_test)  # Apply the same scaling to test set

# Train the classifier (SVC with linear kernel)
cl1 = SVC(kernel='linear', random_state=0)
cl1.fit(X_train, y_train)

# Make predictions
y_predict = cl1.predict(X_test)
print("Predictions:", y_predict)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy
acc = accuracy_score(y_test, y_predict)
print("Accuracy:", acc)
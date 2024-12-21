import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv(r"C:\Users\MSI\Downloads\ParisHousingClass.csv")

# Feature and target selection
x = data.iloc[:, [2, 3]].values  # Select features
y = data.iloc[:, -1].values      # Select target (flattened to 1D)

# Display the feature set (x)
print(x)

# Initialize the Naive Bayes model
model = GaussianNB()

# Fit the model
model.fit(x, y)

# Make predictions
res = model.predict(x)

# Calculate accuracy
acc = accuracy_score(y, res)

# Display results
print("Predictions:", res)
print("Accuracy:", acc)
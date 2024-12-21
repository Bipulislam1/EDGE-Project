import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# a) Load the dataset into a dataframe
data = pd.read_csv(r"C:\Users\MSI\Desktop\Q-14\Final_Xm_salary_dataset.csv")  # Replace with the actual file path
print("First 5 rows of the dataset:")
print(data.head())

# b) Handle empty or categorical data from the dataset
# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# If there are missing values in the columns, we can handle them
# Example: fill missing values in 'Salary' or 'YearsExperience' with the mean or median
data['YearsExperience'] = data['YearsExperience'].fillna(data['YearsExperience'].mean())
data['Salary'] = data['Salary'].fillna(data['Salary'].mean())

# If there are any categorical columns, we need to encode them
# For example, if there were a 'Gender' column, we would do the following:
# data['Gender'] = pd.get_dummies(data['Gender'], drop_first=True)  # If needed

# c) Implement a machine learning model which can predict the salary
# Select features (X) and target variable (y)
X = data[['Age', 'YearsExperience']]  # Features
y = data['Salary']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# d) Plot and visualize the performance of the model over test data
# Scatter plot of actual vs predicted salary
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.title("Actual vs Predicted Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.show()

# Plot a residual plot to check the model performance
sns.residplot(x=y_test, y=y_pred, lowess=True, line_kws={'color': 'red'})
plt.title("Residual Plot")
plt.xlabel("Actual Salary")
plt.ylabel("Residuals")
plt.show()

# Optionally, plot a line of best fit
plt.figure(figsize=(8, 6))
plt.scatter(X_test['YearsExperience'], y_test, color='blue', label="Actual Salary")
plt.scatter(X_test['YearsExperience'], y_pred, color='red', label="Predicted Salary")
plt.title("Actual vs Predicted Salary with Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# a) Load the dataset and display the first 5 rows
data = pd.read_csv(r"C:\Users\MSI\Downloads\Final_exam_data.csv")  # Replace with the actual file path
print("First 5 rows of the dataset:")
print(data.head())

# b) Pre-processing the data
# Handle categorical data (e.g., Gender)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Convert Male/Female to 1/0

# Select features and target variable
X = data[['Gender', 'Age', 'Salary']]  # Features
y = data['Purchased']  # Corrected target variable

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# c) Develop and evaluate the model
# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

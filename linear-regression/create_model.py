import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv("titanic.csv")

# Select features
features = ["Pclass", "Sex", "Age", "Fare"]
data = data[features + ["Survived"]]

# Convert categorical data
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Fill missing values
data["Age"].fillna(data["Age"].median(), inplace=True)

# Define inputs and target
X = data[features]
y = data["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 2. Save the model to a file
joblib.dump(model, 'titanic_model.pkl')

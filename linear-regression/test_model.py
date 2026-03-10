import joblib
import pandas as pd

# 1. Load the model
model = joblib.load('titanic_model.pkl')

# 2. Create a test passenger (Pclass: 3, Sex: 0 for male, Age: 22, Fare: 7.25)
test_passenger = pd.DataFrame([[3, 0, 22, 7.25]], columns=["Pclass", "Sex", "Age", "Fare"])

# 3. Predict (returns [0] for Died or [1] for Survived)
prediction = model.predict(test_passenger)

print(f"Prediction: {'Survived' if prediction[0] == 1 else 'Died'}")

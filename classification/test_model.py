import cv2
import joblib
import numpy as np

# 1. Load the saved model (Ensure 'pip install joblib' was run)
model_file = 'sticky_note_model.pkl'

try:
    model = joblib.load(model_file)
    print(f"Successfully loaded: {model_file}")
except FileNotFoundError:
    print("Error: Model file not found. Run the training script first.")
    exit()

# 2. Function to test an unseen image
def identify_label(image_path):
    # Load the image as grayscale (consistent with training set construction)
    img = cv2.imread(image_path, 0)
    
    if img is not None:
        # Step: Preprocessing (Resize to 64x64 and flatten to match training data)
        resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
        
        # Step: Predict the identity (the name or plate)
        prediction = model.predict(resized)
        return prediction
    else:
        return "Image file not found at path."

# 3. Test with a specific image
# Provide the path to a PNG file NOT used in the training folder
test_image = 'test/sample.png' 
result = identify_label(test_image)

print(f"The Machine Learning model identifies the label as: {result}")
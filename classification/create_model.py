# pip install scikit-learn  numpy  joblib  

import cv2, os, joblib
import numpy as np
from sklearn.svm import SVC

X, y = [], []
root_dir = 'data' 

# 1. Training Set Construction [10 marks]
if os.path.exists(root_dir):
    for label_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label_name)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.lower().endswith('.png'):
                    img = cv2.imread(os.path.join(folder_path, f), 0)
                    if img is not None:
                        X.append(cv2.resize(img, (64, 64)).flatten())
                        y.append(label_name)

# 2. Implementation: Only train and save if data was found [20 marks]
if X:
    model = SVC(kernel='linear').fit(X, y) # This defines 'model'
    print("Training complete.")
    
    # Save the model so it can be used for the airport cameras later
    joblib.dump(model, 'sticky_note_model.pkl')
    print("Model saved as sticky_note_model.pkl")
else:
    print("Error: No images found. 'model' was never created.")
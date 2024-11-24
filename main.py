import argparse
import pandas as pd
import pickle
import numpy as np
from CustomPreprocessor import CustomPreprocessor  # Import your custom preprocessor

# Load pre-trained models
def load_models():
    try:
        with open('SVM_Solo_woGans_convert.pkl', 'rb') as svm_file:
            svm_model = pickle.load(svm_file)
        print("SVM model loaded successfully.")
        
        with open('RANDOM FOREST_GANS_25 convert.pkl', 'rb') as rf_file:
            rf_model = pickle.load(rf_file)
        print("Random Forest model loaded successfully.")
        
        with open('stacking_meta_model.pkl', 'rb') as meta_file:
            meta_model = pickle.load(meta_file)
        print("Stacking model loaded successfully.")
        
        return svm_model, rf_model, meta_model
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()

# Predict using base and stacking models
def predict(svm_model, rf_model, meta_model, X):
    try:
        # Generate predictions from the base learners
        svm_preds = svm_model.predict(X).reshape(-1, 1)
        rf_preds = rf_model.predict(X).reshape(-1, 1)
        
        # Combine predictions for the stacking model
        stacked_features = np.hstack((rf_preds, svm_preds))
        final_preds = meta_model.predict(stacked_features)
        return final_preds
    except Exception as e:
        print(f"Error during prediction: {e}")
        exit()

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Student Performance Prediction Application")
    parser.add_argument('--data', type=str, required=True, help="Path to the raw dataset for prediction.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the prediction results.")
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    svm_model, rf_model, meta_model = load_models()
    
    # Load raw dataset
    try:
        raw_data = pd.read_csv(args.data)
        print(f"Raw data loaded successfully from {args.data}.")
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        exit()
    
    # Preprocess the data
    print("Preprocessing the data...")
    preprocessor = CustomPreprocessor()
    X = preprocessor.fit_transform(raw_data)
    
    # Make predictions
    print("Making predictions...")
    predictions = predict(svm_model, rf_model, meta_model, X)
    
    # Save predictions to file
    print(f"Saving predictions to {args.output}...")
    pd.DataFrame(predictions, columns=["Predicted_Status"]).to_csv(args.output, index=False)
    print("Predictions saved successfully.")

if __name__ == "__main__":
    main()

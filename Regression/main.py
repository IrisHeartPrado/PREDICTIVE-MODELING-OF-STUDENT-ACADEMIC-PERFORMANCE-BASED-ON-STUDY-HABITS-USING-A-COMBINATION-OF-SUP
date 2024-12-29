import pandas as pd
import pickle
import numpy as np
from CustomPreprocessor import CustomPreprocessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse

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
        sys.exit()

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
        sys.exit()

# Evaluate model performance
def evaluate(true_labels_path, predictions):
    try:
        true_labels = pd.read_csv(true_labels_path)['Status']
        
        # Map labels if necessary
        label_mapping = {'Regular': 1, 'Irregular': 0}  
        true_labels = true_labels.replace(label_mapping).values  
        
        # Accuracy
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy on new data: {accuracy:.3f}")
        
        # Classification Report
        report = classification_report(true_labels, predictions)
        print("Classification Report:\n", report)
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Irregular', 'Regular'])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix: Predictions vs True Labels")
        plt.show()
        
        # Comparison dataframe for analysis
        comparison_df = pd.DataFrame({
            'Index': np.arange(len(true_labels)),
            'True Labels': true_labels,
            'Predictions': predictions
        })

        comparison_df['Correct'] = comparison_df['True Labels'] == comparison_df['Predictions']

        # Count correct and wrong predictions
        summary = comparison_df['Correct'].value_counts().reset_index()
        summary.columns = ['Correct', 'Count']
        summary['Correct'] = summary['Correct'].map({True: 'Correct', False: 'Wrong'})

        # Plot the results
        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x='Correct', y='Count', palette=['green', 'red'])
        plt.title("Number of Correct and Wrong Predictions")
        plt.xlabel("Prediction Type")
        plt.ylabel("Count")
        plt.show()
    except Exception as e:
        print(f"Error during evaluation: {e}")

# Main function that handles argument parsing
def main():
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('--data', required=True, help='Path to the dataset')
    parser.add_argument('--output', required=True, help='Path to save predictions')
    parser.add_argument('--true_labels', help='Path to true labels for evaluation')
    args = vars(parser.parse_args())  # Convert parsed arguments into a dictionary

    # Load models
    print("Loading models...")
    svm_model, rf_model, meta_model = load_models()
    
    # Load raw dataset
    try:
        raw_data = pd.read_csv(args['data'])
        print(f"Raw data loaded successfully from {args['data']}.")
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        sys.exit()
    
    # Preprocess the data
    print("Preprocessing the data...")
    preprocessor = CustomPreprocessor()
    X = preprocessor.fit_transform(raw_data)
    
    # Make predictions
    print("Making predictions...")
    predictions = predict(svm_model, rf_model, meta_model, X)
    
    # Save predictions to file
    print(f"Saving predictions to {args['output']}...")
    pd.DataFrame(predictions, columns=["Predicted_Status"]).to_csv(args['output'], index=False)
    print("Predictions saved successfully.")
    
    # Evaluate if true labels are provided
    if 'true_labels' in args:
        print("Evaluating predictions...")
        evaluate(args['true_labels'], predictions)

if __name__ == "__main__":
    main()

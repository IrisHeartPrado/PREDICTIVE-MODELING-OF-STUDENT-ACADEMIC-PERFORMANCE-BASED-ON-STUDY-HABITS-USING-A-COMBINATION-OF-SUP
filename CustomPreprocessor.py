from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
import numpy as np

random.seed(25)
np.random.seed(25)

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.features_to_impute = [
            'Homework', 'Time_Allocation', 'Reading_and_Note_Taking', 
            'Study_Period_Procedures', 'Examination', 'Teachers_Consultation'
        ]
        self.columns_to_drop = [
            'Student Number', 'Name', 'Subject_1', 'Subject_2',
            'Subject_3', 'Subject_4', 'Subject_5', 'Subject_6', 
            'Subject_7', 'Subject_8'
        ]
        self.categorical_cols = ['Time_Allocation', 'Study_Period_Procedures']
        self.num_cols = None

    def fit(self, X, y=None):
        print("Fitting the custom preprocessor...")
        X = X.copy()

        # Drop unnecessary columns
        X = X.drop(columns=self.columns_to_drop, errors='ignore')

        # Rename columns
        X = X.rename(columns={ 
            "Final Grade": "Final_Grade",
            "Subjects Failed": "Subjects_Failed",
            "Time Allocation": "Time_Allocation",
            "Reading and Note Taking": "Reading_and_Note_Taking",
            "Study Period Procedures": "Study_Period_Procedures",
            "Teachers Consultation": "Teachers_Consultation",
        })

        # Impute missing values based on value distribution
        for feature in self.features_to_impute:
            if feature in X.columns:
                print(f"Calculating value distribution for {feature}...")
                feature_counts = X[feature].dropna().value_counts(normalize=True)
                X[feature] = X[feature].apply(
                    lambda x: random.choices(
                        feature_counts.index, 
                        weights=feature_counts.values, 
                        k=1
                    )[0] if pd.isnull(x) else x
                )

        # Fit LabelEncoders for categorical features
        for col in ['Homework', 'Reading_and_Note_Taking', 'Teachers_Consultation', 'Status']:
            if col in X.columns:
                print(f"Fitting LabelEncoder for {col}...")
                le = LabelEncoder()
                X[col] = X[col].fillna("Unknown")  # Fill missing values before encoding
                le.fit(X[col])
                self.label_encoders[col] = le

        # Handle one-hot encoding for categorical columns
        if any(col in X.columns for col in self.categorical_cols):
            X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)

        # Save numeric columns (no standardization)
        self.num_cols = X.select_dtypes(include=['float64', 'int']).columns

        return self

    def transform(self, X):
        print("Transforming the data...")
        X = X.copy()

        # Drop unnecessary columns
        X = X.drop(columns=self.columns_to_drop, errors='ignore')

        # Rename columns
        X = X.rename(columns={ 
            "Final Grade": "Final_Grade",
            "Subjects Failed": "Subjects_Failed",
            "Time Allocation": "Time_Allocation",
            "Reading and Note Taking": "Reading_and_Note_Taking",
            "Study Period Procedures": "Study_Period_Procedures",
            "Teachers Consultation": "Teachers_Consultation",
        })

        # Impute missing values based on value distribution
        for feature in self.features_to_impute:
            if feature in X.columns:
                print(f"Imputing missing values for {feature}...")
                feature_counts = X[feature].dropna().value_counts(normalize=True)
                X[feature] = X[feature].apply(
                    lambda x: random.choices(
                        feature_counts.index, 
                        weights=feature_counts.values, 
                        k=1
                    )[0] if pd.isnull(x) else x
                )

        # Apply LabelEncoders to categorical features
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna("Unknown")
                X[col] = le.transform(X[col])

        # Handle one-hot encoding for categorical columns
        if any(col in X.columns for col in self.categorical_cols):
            X = pd.get_dummies(X, columns=self.categorical_cols)
            one_hot_cols = [col for col in X.columns if col.startswith(tuple(self.categorical_cols))]
            X[one_hot_cols] = X[one_hot_cols].astype(int)

        return X
    

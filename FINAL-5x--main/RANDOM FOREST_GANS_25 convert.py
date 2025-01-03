#!/usr/bin/env python
# coding: utf-8

# Import Necessary Libraries 

# In[1]:


import pandas as pd   
import numpy as np 
import seaborn as sns 
import matplotlib.pylab as plt 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_columns',200)

# Handling Missing Values 
from sklearn.impute import SimpleImputer

# SMOTE 
from imblearn.over_sampling import SMOTE

# Standarlization 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# For pipeline to avoid data leakage 
from imblearn.pipeline import Pipeline as ImbPipeline

# Cross Validation purposes 
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV

)

# Training Model 
from sklearn.ensemble import RandomForestClassifier

# Evaluator for testing 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay


# Importing functions that will generate synthetic data 
# 
# Student contains name, year, student number.  
# GrdSystem contains the individual grades per subject, the calculator for computing grades (Based on student handbook), and the indicator of the student status (Regular or Irregular).    
# S_H_Survey will give the answers: 
# 
# Homework/Assignments - Do you feel that completing your homework helps you understand the course material better? Yes/No
# 
# Time Allocation - Self-reported ratings on ability to manage study time (e.g., poor, fair, good).
# 
# Reading and Note-taking - Types of note-taking methods used (e.g., handwritten, digital).
# 
# Study Period Procedures - Use of Study Techniques: Frequency of using different study techniques (e.g., summarizing, group discussions, flashcards, pomodoro).
# 
# Examination Taking - Average time spent preparing for exams.
# 
# Teachers Consultation  - Do you regularly consult with your teachers outside of class? Yes/No 

# In[2]:


from stdInfo_Function import Student

student = Student(624)
student_info = student.std_info_dt()
print(student_info)


# In[3]:


from stdGrade_Function import GrdSystem

students = GrdSystem(624)
student_grades = students.overall_dt_stdGrades()
print(student_grades)


# In[4]:


from StudyHbtsSurvey_Function import S_H_Survey

survey = S_H_Survey(624, total_respondents= 624, respondents_ans=531)
student_survey = survey.std_info_and_survey()
print(student_survey)


# Concatenating Dataframes 

# In[5]:


concat_data = pd.merge(student_grades, student_survey, how= "left", on= ["Student Number", "Name", "Year"])
concat_data


# Understanding dataframes 

# In[6]:


concat_data.shape


# In[7]:


concat_data.dtypes


# In[8]:


concat_data.describe()


# Cleaning the Data

# In[9]:


#Checking Duplicates 

concat_data.duplicated().sum()


# In[10]:


#Checking null values 

concat_data.isnull().sum()


# In[11]:


#Dropping unnecessary columns 

concat_data.columns 


# In[12]:


concat_data = concat_data[[
    #    'Name', 
        'Year', 
    #    'Student Number', 
    #    'Subject_1', 'Subject_2', 'Subject_3',
    #    'Subject_4', 'Subject_5', 'Status', 'Subject_6', 'Subject_7',
    #    'Subject_8', 
       'Final Grade', 'Subjects Failed', 'Homework',
       'Time Allocation', 'Reading and Note Taking', 'Study Period Procedures',
       'Examination', 'Teachers Consultation', 'Status']].copy()

concat_data.head()


# In[13]:


# Renaming Columns 

concat_data = concat_data.rename(columns = { 
    "Final Grade" : "Final_Grade",
    "Subjects Failed" : "Subjects_Failed",
    "Time Allocation" : "Time_Allocation",
    "Reading and Note Taking" : "Reading_and_Note_Taking",
    "Study Period Procedures" : "Study_Period_Procedures",
    "Teachers Consultation": "Teachers_Consultation",
})

concat_data.head()


# In[14]:


#Converting dtypes float to int 

concat_data['Final_Grade'] = pd.to_numeric(concat_data['Final_Grade'], errors='coerce')

concat_data.dtypes


# Handling Missing Values 

# In[15]:


concat_data.isnull().sum()


# In[16]:


sns.heatmap(concat_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[17]:


sns.set_style('whitegrid')
sns.countplot(x='Status', hue='Homework', data=concat_data, palette='Paired')


# In[18]:


sns.set_style('whitegrid')
sns.countplot(x='Status', hue='Time_Allocation', data=concat_data, palette='Paired')


# In[19]:


sns.set_style('whitegrid')
sns.countplot(x='Status', hue='Reading_and_Note_Taking', data=concat_data, palette='Paired')


# In[20]:


sns.set_style('whitegrid')
sns.countplot(x='Status', hue='Study_Period_Procedures', data=concat_data, palette='Paired')


# In[21]:


sns.set_style('whitegrid')
sns.countplot(x='Status', hue='Examination', data=concat_data, palette='Paired')


# In[22]:


sns.set_style('whitegrid')
sns.countplot(x='Status', hue='Teachers_Consultation', data=concat_data, palette='Paired')


# In[23]:


import random

homework_counts = concat_data['Homework'].dropna().value_counts(normalize=True)

# Fill missing values in 'Homework' based on the distribution of existing values
concat_data['Homework'] = concat_data['Homework'].apply(
    lambda x: random.choices(homework_counts.index, 
                             weights=homework_counts, 
                             k=1)[0] if pd.isnull(x) else x
)


# In[24]:


Time_Allocation_counts = concat_data['Time_Allocation'].dropna().value_counts(normalize=True)

# Fill missing values in 'Time_Allocation' based on the distribution of existing values
concat_data['Time_Allocation'] = concat_data['Time_Allocation'].apply(
    lambda x: random.choices(Time_Allocation_counts.index, 
                             weights=Time_Allocation_counts, 
                             k=1)[0] if pd.isnull(x) else x
)


# In[25]:


Reading_and_Note_Taking_counts = concat_data['Reading_and_Note_Taking'].dropna().value_counts(normalize=True)

# Fill missing values in 'Reading_and_Note_Taking' based on the distribution of existing values
concat_data['Reading_and_Note_Taking'] = concat_data['Reading_and_Note_Taking'].apply(
    lambda x: random.choices(Reading_and_Note_Taking_counts.index, 
                             weights=Reading_and_Note_Taking_counts, 
                             k=1)[0] if pd.isnull(x) else x
)


# In[26]:


Study_Period_Procedures_counts = concat_data['Study_Period_Procedures'].dropna().value_counts(normalize=True)

# Fill missing values in 'Study_Period_Procedures' based on the distribution of existing values
concat_data['Study_Period_Procedures'] = concat_data['Study_Period_Procedures'].apply(
    lambda x: random.choices(Study_Period_Procedures_counts.index, 
                             weights=Study_Period_Procedures_counts, 
                             k=1)[0] if pd.isnull(x) else x
)


# In[27]:


Examination_counts = concat_data['Examination'].dropna().value_counts(normalize=True)

# Fill missing values in 'Examination' based on the distribution of existing values
concat_data['Examination'] = concat_data['Examination'].apply(
    lambda x: random.choices(Examination_counts.index, 
                             weights=Examination_counts, 
                             k=1)[0] if pd.isnull(x) else x
)


# In[28]:


Teachers_Consultation_counts = concat_data['Teachers_Consultation'].dropna().value_counts(normalize=True)

# Fill missing values in 'Teachers_Consultation' based on the distribution of existing values
concat_data['Teachers_Consultation'] = concat_data['Teachers_Consultation'].apply(
    lambda x: random.choices(Teachers_Consultation_counts.index, 
                             weights=Teachers_Consultation_counts, 
                             k=1)[0] if pd.isnull(x) else x
)


# In[29]:


concat_data


# In[30]:


concat_data.isnull().sum()


# Some data are in object, we need to convert it to numerical for better understading 

# In[31]:


# Converting using pd.dummies

features_dummies = pd.get_dummies(concat_data[['Time_Allocation','Study_Period_Procedures']])

features_dummies = features_dummies.astype(int)

features_dummies


# In[32]:


# Converting using labelencoder 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

concat_data['Homework'] = label_encoder.fit_transform(concat_data['Homework'])
concat_data['Reading_and_Note_Taking'] = label_encoder.fit_transform(concat_data['Reading_and_Note_Taking'])
concat_data['Teachers_Consultation'] = label_encoder.fit_transform(concat_data['Teachers_Consultation'])
concat_data['Status'] = label_encoder.fit_transform(concat_data['Status'])

concat_data


# In[33]:


# Concatenate the dummies with the original data
concat_data_updated = pd.concat([concat_data, features_dummies], axis=1)

# Drop the original columns 
concat_data_updated = concat_data_updated.drop(['Time_Allocation', 'Study_Period_Procedures'], axis=1)

concat_data_updated.head()


# In[34]:


concat_data_updated.columns


# In[35]:


concat_data_updated = concat_data_updated[['Year', 'Final_Grade', 'Subjects_Failed', 'Homework',
       'Reading_and_Note_Taking', 'Examination', 'Teachers_Consultation',
        'Time_Allocation_Fair', 'Time_Allocation_Good',
       'Time_Allocation_Poor', 'Study_Period_Procedures_Flashcards',
       'Study_Period_Procedures_Group Discussions',
       'Study_Period_Procedures_Pomodoro',
       'Study_Period_Procedures_Summarizing','Status']]

concat_data_updated


# In[36]:


concat_data_updated.dtypes


# GANS

# In[37]:


import ctgan.synthesizers.ctgan as ctgan_mod
print(dir(ctgan_mod))


# In[38]:


from ctgan import CTGAN

ctgan = CTGAN(
    generator_dim=(512, 512, 256),
    discriminator_dim=(512, 256, 128),
    epochs=3000,
    batch_size=32,
    verbose=True,
    pac=4,
)




# In[39]:


# Preprocessing Continuous Features
from scipy.stats import skew

scaler = MinMaxScaler(feature_range=(0, 1))
continuous_features = ['Final_Grade', 'Subjects_Failed', 'Examination']

for col in continuous_features:
    if skew(concat_data_updated[col]) > 1:  # Log-transform skewed features
        concat_data_updated[col] = np.log1p(concat_data_updated[col])

concat_data_updated[continuous_features] = scaler.fit_transform(concat_data_updated[continuous_features])

print(concat_data_updated[continuous_features].describe())


# In[40]:


concat_data_updated['Final_Grade_Binned'] = pd.cut(
    concat_data_updated['Final_Grade'],
    bins=[0, 0.5, 0.7, 1.0], 
    labels=['Low', 'Medium', 'High']
)

concat_data_updated['Subjects_Failed_Binned'] = pd.cut(
    concat_data_updated['Subjects_Failed'],
    bins=[-1, 0.2, 0.5, 1.0], 
    labels=['None', 'Few', 'Many']
)


# In[41]:


status_counts = concat_data_updated['Status'].value_counts()

n_irregular = status_counts[0]
n_regular = status_counts[1]

print(f"Regular Students: {n_regular}, Irregular Students: {n_irregular}")


# In[42]:


# Train the CTGAN Model

discrete_columns = ['Status', 'Time_Allocation_Fair', 'Time_Allocation_Good', 'Time_Allocation_Poor',
                    'Study_Period_Procedures_Flashcards', 'Study_Period_Procedures_Group Discussions',
                    'Study_Period_Procedures_Pomodoro', 'Study_Period_Procedures_Summarizing', 'Final_Grade_Binned',
                    'Subjects_Failed_Binned']

ctgan.fit(concat_data_updated, discrete_columns=discrete_columns)


# In[43]:


# Generate Synthetic Data
synthetic_data = ctgan.sample(n=1248)


# In[44]:


# Post-processing: Fix extreme values for "Examination"
synthetic_data['Examination'] = synthetic_data['Examination'].clip(
    lower=concat_data_updated['Examination'].min(),
    upper=concat_data_updated['Examination'].max()
)

# Additional post-processing (if needed)
for column in continuous_features:  # Adjust distributions
    synthetic_data[column] = synthetic_data[column].clip(
        lower=concat_data_updated[column].min(),
        upper=concat_data_updated[column].max()
    )

# Validate synthetic data
synthetic_data.describe()


# In[54]:


# Drop Binned Columns
binned_columns = ['Final_Grade_Binned', 'Subjects_Failed_Binned']

synthetic_data.drop(columns=binned_columns, inplace=True)

synthetic_data


# In[56]:


#Visualization for the distribution of numerical features between the original and synthetic datasets.
import matplotlib.pyplot as plt
import seaborn as sns

numerical_columns = ['Final_Grade', 'Subjects_Failed', 'Homework']

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(concat_data_updated[col], label='Original', shade=True)
    sns.kdeplot(synthetic_data[col], label='Synthetic', shade=True)
    plt.title(f'Distribution of {col}')
    plt.legend()
    plt.show()


# In[57]:


#Visualization for the distribution of categorical features between the original and synthetic datasets.
categorical_columns = ['Status', 'Time_Allocation_Fair', 'Time_Allocation_Good']

for col in categorical_columns:
    plt.figure(figsize=(6, 4))
    original_counts = concat_data_updated[col].value_counts()
    synthetic_counts = synthetic_data[col].value_counts()

    bar_width = 0.35
    index = range(len(original_counts))

    plt.bar(index, original_counts.values, bar_width, label='Original', alpha=0.7)
    plt.bar([i + bar_width for i in index], synthetic_counts.values, bar_width, label='Synthetic', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xticks([i + bar_width / 2 for i in index], original_counts.index, rotation=45)
    plt.legend()
    plt.show()


# In[58]:


# Summary statistics comparison
print("Original Data Statistics:")
print(concat_data_updated.describe())

print("\nSynthetic Data Statistics:")
print(synthetic_data.describe())


# In[59]:


import pandas as pd

#Combining the original and synthetic data 
combined_data = pd.concat([concat_data_updated, synthetic_data], ignore_index=True)


# In[60]:


from sklearn.utils import shuffle

# Shuffle the combined dataset to ensure it doesn’t have patterns based on the original vs. synthetic data split.
combined_data = shuffle(combined_data, random_state=42)


# Independent variables are the predictor or the features that will help us predict whether the student has probability to be a irregular student for the next semester or will probably stay as regular student based on their study habits that is gathered through the survey and final grades

# Dependent variable is the target variable or what we are trying to know which is the status of the student

# In[61]:


# Dependent(y) and Independent(x) Variable segregation 

x = combined_data.drop(['Status'], axis=1)
y = combined_data['Status']


# In[62]:


#Data Spliting (Training and Testing Set)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)


# In[63]:


y_train_df = y_train.rename("Target")

# Concatenate the feature DataFrame (x_train) with the target column (y_train_df)
columnStatus_for_correlation = pd.concat([x_train, y_train_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = columnStatus_for_correlation.corr()


# In[50]:


correlation_matrix


# In[51]:


plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', annot_kws={"size": 5})
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.yticks(rotation=0)   # Rotate y-axis labels
plt.show()


# TRAINING RANDOM FOREST 

# In[52]:


param_grid = {
    'n_estimators': [100, 150, 200],          
    'bootstrap': [True, False],              
    'max_depth': [8, 10, 12],               
    'min_samples_split': [8, 10, 12],        
    'class_weight': [
        {0: 1, 1: 5}, 
        {0: 1, 1: 8}, 
        {0: 1, 1: 10}, 
        'balanced'
    ],  
}

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the model
rf_model = RandomForestClassifier(random_state=25)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='f1',        
    cv=5,                
    verbose=2,          
    n_jobs=-1            
)

# Fit the model on training data
grid_search.fit(x_train, y_train)

# Display best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)


# In[53]:


from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import ADASYN

rf_model = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    max_depth=6, 
    min_samples_split=10,
    min_samples_leaf= 15,
    class_weight={0: 0.692, 1: 10},
    random_state=25
)

pipeline_rf = ImbPipeline([
    ('scaler', StandardScaler()),
    ('Class Imbalance', SMOTE(random_state=25)),
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),
    ('classifier', rf_model)
])

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)

cv_scores = cross_val_score(pipeline_rf, x_train, y_train, cv=kf, scoring='f1')

print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', np.mean(cv_scores))
print('Standard deviation of cross-validation score:', np.std(cv_scores))


# In[54]:


pipeline_rf.fit(x_train, y_train)

y_pred_rf = pipeline_rf.predict(x_test)

test_accuracy = accuracy_score(y_test, y_pred_rf)
print('Test Accuracy:', test_accuracy)

print('Classification Report on Test Set:')
print(classification_report(y_test, y_pred_rf))


# In[58]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook file
with open('RANDOM FOREST_GANS_25 convert copy 4.ipynb') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
python_script, _ = exporter.from_notebook_node(notebook_content)

# Save the Python script to a .py file
with open('RANDOM FOREST_GANS_25 convert.py', 'w') as f:
    f.write(python_script)


# In[59]:


import pickle

with open('RANDOM FOREST_GANS_25 convert.pkl', 'wb') as f:
    pickle.dump(pipeline_rf, f)


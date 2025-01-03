#!/usr/bin/env python
# coding: utf-8

# Import Necessary Libraries 

# In[1]:


import pandas as pd # To handle and manipulate structured data
import numpy as np  # To perform numerical operations efficiently, especially for arrays and matrices.
import seaborn as sns # To create advanced and aesthetically pleasing data visualizations based on Matplotlib.
import matplotlib.pylab as plt # To provide functionality for creating basic plots and customizing visualizations.
import matplotlib.pyplot as plt
plt.style.use('ggplot') #Styling
pd.set_option('display.max_columns',200) # To show up to 200 columns when displaying DataFrames.

# Handling Imbalance data
from imblearn.over_sampling import ADASYN, SMOTE

# Standarlization 
from sklearn.preprocessing import StandardScaler

# For pipeline to avoid data leakage 
from imblearn.pipeline import Pipeline as ImbPipeline


from sklearn.model_selection import (
    train_test_split, # To split the dataset into training and testing sets
    cross_val_score, # It estimates the model's performance compared to a single train-test split.
    StratifiedKFold, # It preserves the proportion of classes in the target variable across all folds.
    GridSearchCV # To find the best combination of hyperparameters
)

# Training Model 
from sklearn.svm import SVC

# Evaluator for testing 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay

import pickle


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


# In[ ]:


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


# In[ ]:


# Converting using labelencoder 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

concat_data['Homework'] = label_encoder.fit_transform(concat_data['Homework'])
concat_data['Reading_and_Note_Taking'] = label_encoder.fit_transform(concat_data['Reading_and_Note_Taking'])
concat_data['Teachers_Consultation'] = label_encoder.fit_transform(concat_data['Teachers_Consultation'])
concat_data['Status'] = label_encoder.fit_transform(concat_data['Status'])

concat_data


# In[ ]:


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


# Independent variables are the predictor or the features that will help us predict whether the student has probability to be a irregular student for the next semester or will probably stay as regular student based on their study habits that is gathered through the survey and final grades

# Dependent variable is the target variable or what we are trying to know which is the status of the student

# In[37]:


# Dependent(y) and Independent(x) Variable segregation 

x = concat_data_updated.drop(['Status'], axis=1)
y = concat_data_updated['Status']


# In[38]:


#Data Spliting (Training and Testing Set)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)


# In[39]:


y_train_df = y_train.rename("Target")

# Concatenate the feature DataFrame (x_train) with the target column (y_train_df)
columnStatus_for_correlation = pd.concat([x_train, y_train_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = columnStatus_for_correlation.corr()


# In[40]:


correlation_matrix


# In[41]:


plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', annot_kws={"size": 5})
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.yticks(rotation=0)   # Rotate y-axis labels
plt.show()


# TRAINING SVM MODEL

# Creating pipeline to avoid data leakage wherein: 
# 
# Standardization, Smote, Model (SVM) are being performed 

# In[ ]:


smote = ADASYN(random_state=25)

svm_model = SVC(C= 0.1, gamma= 0.2, kernel='rbf', class_weight= {0: 5, 1: 10}, probability=True) #0: 0.900, 1: 1

pipeline_svm = ImbPipeline(steps = [('scaler', StandardScaler()),
                                ('smote', smote),
                                ('classifier', svm_model)
                                ])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)

cv_scores = cross_val_score(pipeline_svm, x_train, y_train, cv=kf, scoring='f1')

print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', np.mean(cv_scores))
print('Standard deviation of cross-validation score:', np.std(cv_scores))


# In[43]:


from sklearn.inspection import permutation_importance

pipeline_svm.fit(x_train, y_train)

perm_importance = permutation_importance(pipeline_svm, x_test, y_test, n_repeats=10, random_state=25)

importance_scores = perm_importance.importances_mean

# Sort the features by importance
sorted_idx = np.argsort(importance_scores)[::-1]

for idx in sorted_idx:
    print(f"Feature: {x_train.columns[idx]}, Importance: {importance_scores[idx]}")


# In[45]:


from sklearn.inspection import permutation_importance

pipeline_svm.fit(x_train, y_train)

perm_importance = permutation_importance(pipeline_svm, x_test, y_test, n_repeats=10, random_state=25)

importance_scores = perm_importance.importances_mean

# Sort the features by importance
sorted_idx = np.argsort(importance_scores)[::-1]

for idx in sorted_idx:
    print(f"Feature: {x_train.columns[idx]}, Importance: {importance_scores[idx]}")


# In[ ]:


pipeline_svm.fit(x_train, y_train)

y_pred_svm = pipeline_svm.predict(x_test)

test_accuracy = accuracy_score(y_test, y_pred_svm)
print('Test Accuracy:', test_accuracy)

print('Classification Report on Test Set:')
print(classification_report(y_test, y_pred_svm))


# In[48]:


train_pred_svm = pipeline_svm.predict(x_train)

print('Classification Report on Training Set:')
print(classification_report(y_train, train_pred_svm))


# In[49]:


from sklearn.metrics import confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))


# In[ ]:


svm = SVC(random_state=25)

param_grid_svm = {
    'C': [0.01, 0.02, 0.03],
    'gamma': [0.1, 0.2, 0.3],
    'kernel': ['linear', 'rbf', 'poly']
}

grid = GridSearchCV(svm, param_grid_svm, refit=True, verbose=2, cv=5)
grid.fit(x_train, y_train)

print(grid.best_params_)
print("Best score found: ", grid.best_score_)


# In[51]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

svm = SVC(random_state=25)

param_dist = {
    'C': uniform(0.1, 20),  
    'gamma': uniform(0.1, 1),
    'kernel': ['linear', 'rbf', 'poly']
}

random = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=10, cv=5, random_state=25, verbose=2) 
random.fit(x_train, y_train)

print(random.best_params_)
print("Best score found: ", random.best_score_)


# In[52]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook file
with open('SVM_Solo_woGans_convert.ipynb') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
python_script, _ = exporter.from_notebook_node(notebook_content)

# Save the Python script to a .py file
with open('SVM_Solo_woGans_convert.py', 'w') as f:
    f.write(python_script)


# In[53]:


with open('SVM_Solo_woGans_convert.pkl', 'wb') as f:
    pickle.dump(pipeline_svm, f)


# In[ ]:





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
plt.xticks(rotation=45)  
plt.yticks(rotation=0)   
plt.show()


# In[42]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

seed = 25
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set the latent dimension for noise input to the generator
latent_dim = 100  # Number of dimensions in the random noise vector
input_dim = x_train.shape[1]  # Number of features in your dataset

# Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize the generator and discriminator
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)


# In[43]:


# Training parameters
num_epochs = 10000
batch_size = 100

# Convert x_train to a NumPy array first
X_train_tensor = torch.Tensor(x_train.values)  # Use `.values` to get the underlying NumPy array

for epoch in range(num_epochs):
    # Discriminator training
    real_data = X_train_tensor[torch.randint(0, X_train_tensor.size(0), (batch_size,))]
    real_labels = torch.ones(batch_size, 1)  # Real data labels are 1

    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z)
    fake_labels = torch.zeros(batch_size, 1)  # Fake data labels are 0

    optimizer_D.zero_grad()
    real_loss = criterion(discriminator(real_data), real_labels)
    fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Generator training
    optimizer_G.zero_grad()
    fake_labels = torch.ones(batch_size, 1)  # Trick discriminator by labeling fake data as real
    g_loss = criterion(discriminator(fake_data), fake_labels)
    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")


# In[44]:


#  Wasserstein GAN (WGAN-GP), a variation of GANs, to improve training stability and enforce the 1-Lipschitz constraint on the Discriminator (now often called the Critic in WGANs).
def compute_gradient_penalty(real_data, fake_data, discriminator):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones_like(d_interpolated),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Hyperparameters
gradient_penalty_weight = 10.0 
label_smoothing_value = 0.9 

# Optimizers and schedulers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0004, betas=(0.5, 0.999))
scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=10)
scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=10)

# Training loop with the updated variable name for gradient penalty
for epoch in range(num_epochs):
    # Discriminator update with gradient penalty
    optimizer_D.zero_grad()
    real_data = X_train_tensor[torch.randint(0, X_train_tensor.size(0), (batch_size,))]
    real_data += 0.05 * torch.randn(real_data.size()).to(real_data.device)  # Add noise to real data
    real_labels = torch.full((batch_size, 1), label_smoothing_value).to(real_data.device)  # Label smoothing

    z = torch.randn(batch_size, latent_dim).to(real_data.device)
    fake_data = generator(z)
    fake_data += 0.05 * torch.randn(fake_data.size()).to(fake_data.device)  # Add noise to fake data

    d_loss_real = criterion(discriminator(real_data), real_labels)
    d_loss_fake = criterion(discriminator(fake_data.detach()), torch.zeros(batch_size, 1).to(real_data.device))
    gp = compute_gradient_penalty(real_data, fake_data, discriminator) * gradient_penalty_weight
    d_loss = d_loss_real + d_loss_fake + gp
    d_loss.backward()
    optimizer_D.step()

    # Generator update
    optimizer_G.zero_grad()
    z = torch.randn(batch_size, latent_dim).to(real_data.device)
    fake_data = generator(z)
    g_loss = criterion(discriminator(fake_data), real_labels)  # We want G to fool D
    g_loss.backward()
    optimizer_G.step()

    # Step the schedulers
    scheduler_D.step(d_loss)
    scheduler_G.step(g_loss)

    # Logging the losses and gradient penalty
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Gradient Penalty: {gp.item():.4f}")



# In[45]:


num_synthetic_samples = 2000  # Adjust as needed
z = torch.randn(num_synthetic_samples, latent_dim)
synthetic_samples = generator(z).detach().numpy()


# In[46]:


# Combine real and synthetic samples
X_train_augmented = np.vstack((x_train, synthetic_samples))

y_train_augmented = np.hstack((y_train, np.random.choice(y_train, num_synthetic_samples)))


# TRAINING RANDOM FOREST 

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif

rf_model = RandomForestClassifier(
    n_estimators=100,
    bootstrap=True,
    max_depth=2, 
    min_samples_split=5,
    class_weight={0:1.5, 1:7},
    random_state=25
)

pipeline_rf = ImbPipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),
    ('classifier', rf_model)
])

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)

cv_scores = cross_val_score(pipeline_rf, X_train_augmented, y_train_augmented, cv=kf, scoring='f1')

print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', np.mean(cv_scores))
print('Standard deviation of cross-validation score:', np.std(cv_scores))


# In[ ]:


pipeline_rf.fit(X_train_augmented, y_train_augmented)

y_pred_rf = pipeline_rf.predict(x_test)

test_accuracy = accuracy_score(y_test, y_pred_rf)
print('Test Accuracy:', test_accuracy)

print('Classification Report on Test Set:')
print(classification_report(y_test, y_pred_rf))


# In[ ]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook file
with open('RANDOM FOREST_GANS_25 convert copy.ipynb') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
python_script, _ = exporter.from_notebook_node(notebook_content)

# Save the Python script to a .py file
with open('RANDOM FOREST_GANS_25 convert.py', 'w') as f:
    f.write(python_script)


# In[ ]:


import pickle

with open('RANDOM FOREST_GANS_25 convert.pkl', 'wb') as f:
    pickle.dump(pipeline_rf, f)


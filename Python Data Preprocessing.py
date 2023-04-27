#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
import os

# Get current directory
current_dir = os.getcwd()

# Load data from CSV file
data = pd.read_csv(os.path.join(current_dir, 'data.csv'))

# Replace missing values with the mean
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(data)

# Select the top 10 features
selector = SelectKBest(chi2, k=10)
selected_data = selector.fit_transform(imputed_data, target)

# Print the selected features
print(selected_data)


# SALARY APP

# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas.core.common import random_state
from tensorflow.python.keras import engine
path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_Salary_App"
os.chdir(path)

import warnings
warnings.filterwarnings("ignore")

# Import
df = pd.read_csv('survey_results_public.csv')
df.columns
df.shape

# Select
df = df[["Employment", "Country", "EdLevel", "YearsCodePro", "ConvertedComp"]]
df = df.rename({"ConvertedComp":"Salary"}, axis=1)

# $Salary and NAs
df = df[df["Salary"].notnull()]
df = df.dropna()
df.isnull().sum()

# $Employment
df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()

# $Country
df['Country'].value_counts()
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map = shorten_categories(df.Country.value_counts(), 200)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()

df = df[df['Salary'] <= 250000]
df = df[df['Salary'] >= 10000]
df = df[df['Country'] != 'Other']

# $Education
df['EdLevel'].unique()

def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'
df['EdLevel'] = df['EdLevel'].apply(clean_education)

# $Experience
df['YearsCodePro'].unique()

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)
df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# Vizualisation
fig, ax = plt.subplots(1,1,figsize=(12,8))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

df.shape

# Table with mean and median
table = df.groupby('Country').mean()
table = table.rename(columns={"YearsCodePro" : "YearsCodeProMean", "Salary" : "SalaryMean"})

table = table.join(df.groupby('Country').median()[["YearsCodePro", 'Salary']])
table = table.rename(columns={"YearsCodePro" : "YearsCodeProMedian", "Salary" : "SalaryMedian"})
table

#df.to_excel("survey_clean.xlsx", index=False)

# ---------------------------------------------------------------------------------------------------------------
# Modeling ------------------------------------------------------------------------------------------------------

# Encoding
from sklearn.preprocessing import LabelEncoder
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df['Country'].unique()

from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()

# Train/Test split
X = df[['Country', 'EdLevel', 'YearsCodePro']]
y = df['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, random_state=0) # random forest regression model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)
print('Percentage difference: ', round(outcome.difference_percentage.abs().mean(),2),'%')

from sklearn.metrics import mean_squared_error, mean_absolute_error
print('Mean absolute error: ',round(mean_absolute_error(y_test, y_pred),4))
print('Root mean squared error: ',round(np.sqrt(mean_squared_error(y_test, y_pred)),4))

# Grid Search for Random Forest Regression
from sklearn.model_selection import GridSearchCV
max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth" : max_depth}

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0)
gs = GridSearchCV(model, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y)

model = gs.best_estimator_

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_pred))
print("${:,.02f}".format(error))

X_sample = np.array([["Czech Republic", "Master’s degree", "1"]])
X_sample[:,0] = le_country.transform(X_sample[:,0])
X_sample[:,1] = le_education.transform(X_sample[:,1])
X_sample = X_sample.astype(float)

y_pred = model.predict(X_sample)
print(f'Salary: ${round(y_pred[0])}')

# Save
import pickle
data = {"model": model, "le_country" : le_country, "le_education" : le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

# Load
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

model_loaded = data["model"]
le_country = data["le_country"]
led_education = data["le_education"]

# Pred
y_pred = model_loaded.predict(X_sample)
y_pred

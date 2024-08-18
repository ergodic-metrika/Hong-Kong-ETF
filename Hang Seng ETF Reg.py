# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:03:03 2024

@author: user
"""

#Package loading
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.api as sm
#Read excel file
file_path = r'D:\Derivatives Trading\Hang Seng ETF Data.xlsx'
df = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# Display the first few rows of the DataFrame
print(df.head())

# Display summary statistics
print(df.describe())

# Check for null values
print(df.isnull().sum())


# Calculate daily returns for '7200 HK', '3110 HK', and 'Hang Seng Index'
df['7200 HK Return'] = df['7200 HK'].pct_change()
df['3110 HK Return'] = df['3110 HK'].pct_change()
df['Hang Seng Tech Return'] = df['Hang Seng Tech'].pct_change()
df['Hang Seng Index Return'] = df['Hang Seng Index'].pct_change()

# Display the first few rows to verify the calculations
print(df[['7200 HK Return', '3110 HK Return', 'Hang Seng Tech Return', 'Hang Seng Index Return']].head())

df.fillna(0, inplace=True)  # This fills all NaN values with 0

plt.figure(figsize=(12, 6))
df[['7200 HK Return', '3110 HK Return', 'Hang Seng Tech Return', 'Hang Seng Index Return']].plot(title='Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.grid(True)
plt.show()


# Remove any NaN values that may affect the regression analysis
df = df.dropna()

# Prepare independent variable (add a constant term to allow statsmodels to fit an intercept)
X = sm.add_constant(df['Hang Seng Index Return'])  # Independent variable (Market returns)

# Prepare dependent variables
y_7200 = df['7200 HK Return']  # Dependent variable for 7200 HK
y_3110 = df['3110 HK Return']  # Dependent variable for 3110 HK
y_HSI_Tech=df['Hang Seng Tech Return']  # Dependent variable for Hang Seng Tech

# Run regression for 7200 HK
model_7200 = sm.OLS(y_7200, X)
results_7200 = model_7200.fit()

# Run regression for 3110 HK
model_3110 = sm.OLS(y_3110, X)
results_3110 = model_3110.fit()

# Run regression for Hang Seng Tech
model_HSI_Tech = sm.OLS(y_HSI_Tech, X)
results_HSI_Tech = model_HSI_Tech.fit()

# Print the summary of regression results
print("Regression results for 7200 HK:")
print(results_7200.summary())
print("\nRegression results for 3110 HK:")
print(results_3110.summary())
print("Regression results for 7200 HK:")
print(results_7200.summary())
print("\nRegression results for Hang Seng Tech:")
print(results_HSI_Tech.summary())
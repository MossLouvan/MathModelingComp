import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


# Data
years = [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000]
temperatures_f = [102, 102, 96, 97, 100, 97, 99, 100, 99, 100, 98, 103, 106, 104, 100, 101, 106, 102, 100, 97, 98, 98, 96, 107]
temperatures_c = [38.9, 38.9, 35.6, 36.1, 37.8, 36.1, 37.2, 37.8, 37.2, 37.8, 36.7, 39.4, 41.1, 40.0, 37.8, 38.3, 41.1, 38.9, 37.8, 36.1, 36.7, 36.7, 35.6, 41.7]

# Create a DataFrame
df = pd.DataFrame({'Year': years, 'Temperature (°F)': temperatures_f, 'Temperature (°C)': temperatures_c})

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['Year'], df['Temperature (°F)'], marker='o', linestyle='-', color='r', label='Temperature (°F)')
plt.plot(df['Year'], df['Temperature (°C)'], marker='s', linestyle='--', color='b', label='Temperature (°C)')

# Labels and Title
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('High Temperatures Observed at Memphis International Airport')
plt.legend()
plt.grid()

# Show Plot
plt.savefig("High temp vs year")




# Prepare the data
X = np.array(years).reshape(-1, 1)  # Feature: Year
y = np.array(temperatures_f)  # Target: Temperature (°F)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
future_years = np.array(range(2024, 2030)).reshape(-1, 1)  # Predict for the next 6 years
future_temps = model.predict(future_years)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(df['Year'], df['Temperature (°F)'], marker='o', linestyle='-', color='r', label='Historical Temperature (°F)')
plt.plot(future_years, future_temps, marker='x', linestyle='--', color='g', label='Predicted Temperature (°F)')

# Labels and Title
plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.title('Temperature Prediction at Memphis International Airport')
plt.legend()
plt.grid()

# Show Plot
plt.savefig("Prediction graph")

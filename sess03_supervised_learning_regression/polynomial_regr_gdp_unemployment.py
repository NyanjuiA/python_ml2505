# Python script to analyse and predict the unemployement rate based on GDP
# using polynomial regression

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Get the path to the file with the data to be loaded
file_path = os.path.abspath(os.path.join(os.getcwd(),"..","files","gdp_unemployment.csv"))

# Load the data from te csv file
data = pd.read_csv(file_path)

# reshape the data
year = data['Year'].values.reshape(-1,1)
gdp = data['GDP'].values.reshape(-1,1)
unemployement_rate = data['Unemployment Rate'].values.reshape(-1,1)

# Create the polynomial features
poly = PolynomialFeatures(degree = 2) # You can adjust the degree(squared terms) to suit your needs
X_poly = poly.fit_transform(gdp)

# Fit (train) the model
model = LinearRegression()
model.fit(X_poly, unemployement_rate)

# Predict the unemployment rate for 2020 given a GDP of $ 620 Billion
gdp_2020 = np.array([[620]])
gdp_2020_poly = poly.transform(gdp_2020)
predicted_unemployment = model.predict(gdp_2020_poly)

# Display the predicted unemployment rate/percentage
print(f"The predicted unemployement rate for 2020 @ GDP of $ 620 Billion is:"
      f"{predicted_unemployment[0][0]:.2f}%.")

# Plot the graph
plt.scatter(gdp,unemployement_rate, color='blue',label='Data Points')
gdp_range = np.linspace(290,630,100).reshape(-1,1)
gdp_range_poly = poly.transform(gdp_range)
predicted_rates = model.predict(gdp_range_poly)
plt.plot(gdp_range, predicted_rates, color='red',label='Polynomial Fit')
plt.scatter(gdp_2020, predicted_unemployment,color='orange',label='2020 Prediction',
            s=100,edgecolors='black')

# Plot labels
plt.title('GDP vs. Unemployement Rate')
plt.xlabel('GDP in Billion $')
plt.ylabel('Unemployment Rate %')
plt.legend()
plt.grid()
plt.show()


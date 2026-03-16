# Python script to demonstrate outlier detection and handling using IQR(interquartile Range)

# Import the required modules
import pandas as pd
import numpy as np

# 1. Create a sample dataset
data = {
   'Name': ['Abigail', "Kamau", "Sharlene", 'Diana', "Mueni","Frank","Grace"],
   'Age': [5,35,32,39,45,120,28], # 5 & 120 are outliers
   'Salary': [50000,2500,54000,52000,110000,47000,51000] # '2500' & '110000' are outliers
}

# 2. Convert the above dictionary into a Pandas dataframe and display it
df = pd.DataFrame(data)
print(f"Original Dataset:\n{df}")

# 3. Detect outliers using IQR method for the 'Age' column
Q1 = df['Age'].quantile(0.25) # First quartile (Q1)
Q3 = df['Age'].quantile(0.75) # Third quartile (Q3)
IQR = Q3 - Q1 # Get the interquartile range
# Define the 'Age' outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# TODO: Detect outliers using IQR method for the 'Salary' column

# Identify and display the outliers for the 'Age' column
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print(f"Detected outliers for the 'Age' column:\n{outliers}")

# 4. Handle outliers -> method i) Remove/drop the outliers
df_no_age_outliers = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
# Display the dataset after removing outliers for the 'Age' column
print(f"Dataset after removing outliers for the 'Age' column:\n{df_no_age_outliers}")

# 5. Handle outliers -> method ii) Cap the outliers with boundary values
df['Age'] = np.where(df['Age'] < lower_bound,lower_bound,
                     np.where(df['Age'] > upper_bound,upper_bound,df['Age']))
# Display the dataset after capping outliers for the 'Age' column
print(f"Dataset after capping the outliers for the 'Age' column:\n{df}")
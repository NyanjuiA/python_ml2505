# Python script to demonstrate how to deal with  missing values

# import the required modules
import pandas as pd
import numpy as np

# 1. Create a sample dataset
data = {
   'Name': ['Abigail', "Kamau", "Sharlene", 'Diana', "Mueni"],
   'Age': [24, np.nan, 22, 32, np.nan],
   'City': ["Nakuru", 'Limuru', np.nan, 'Homabay', "Makueni"],
}

# 2. Convert the above dictionary into a dataframe
df = pd.DataFrame(data)

# 3. Display the original datafram with missing values
print(f"Original Dataset with missing values:\n{df}")

# 4. Handle the missing data/values
# Option 1: Drop rows with missing values
df_dropna = df.dropna()
# Display the dataset after dropping rows with missing values
print(f"Dataset after dropping rows with missing values:\n{df_dropna}")

# Option 2: Impute/fill in the missing values
# (Fill in the mean for the 'Age' column)
df.fillna({'Age':df['Age'].mean()},inplace=True)

# Fill in the missing categorical values(forward fill for the 'City' column)
df['City'] = df['City'].ffill()
# Display the dataset fater imputing/fillin in the missing values
print(f"Dataset after imputing/filling in the missing values:\n{df}")
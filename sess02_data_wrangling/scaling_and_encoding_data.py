# Python script to demonstrate data transformation using scaling and encoding
# NB: Ensure that sci-kit learn is installed (pip install scikit-learn)

# Import the required modules
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# 1. Create a sample dataset
data = {
   'Name': ['Abigail', "Kamau", "Sharlene", 'Diana', "Mueni"],
   'Age': [55, 32, 22, 45, 28],
   'Salary': [50000, 54000, 52000, 110000, 51000],
   'Gender': ['Female', "Male", 'Female', 'Female', "Female"],
   'City': ["Nakuru", 'Limuru', 'Kisumu', 'Homabay', "Makueni"],
}

# Convert above dictionary into a Pandas dataframe
df = pd.DataFrame(data)

# 2. Standardise the age and salary
scaler = StandardScaler()
df[['Age','Salary']] = scaler.fit_transform(df[['Age','Salary']])

# 3. One-Hot encoding for Gender and city
encoder = OneHotEncoder(sparse_output=False)
encoded_columns = pd.DataFrame(encoder.fit_transform(df[['Gender','City']]),
                               columns=encoder.get_feature_names_out(['Gender','City']))

# 4. Join/concatenate the transformed data and display it
df = pd.concat([df.drop(columns=['Gender','City']),encoded_columns], axis=1)

print(df)

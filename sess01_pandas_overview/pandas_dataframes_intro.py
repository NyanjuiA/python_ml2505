# Python script/file to demonstrate Pandas DataFrames by demonstrating how to create, display
# and explore them.

# Import the required modules
import pandas as pd
import numpy as np

# 1. Create a Data Frame from a Dictionary
print("---- 1. Create a Data Frame from a Dictionary ----")
data = {
   'Name': ['Abigail',"Kamau","Sharlene", 'Diana',"Mueni"],
   'Age' : [25,30,35, 28,32],
   'City': ["Nakuru", 'Limuru','Kisumu','Homabay',"Makueni"],
   'Salary': [55000, 65000, 72000,48000,60000],
   'Department': ['HR','IT',"IT",'Marketing',"Finance"]
}

# 2. Create and display a Pandas Dataframe from the above data
df = pd.DataFrame(data)
print(f"Employee Details dataFrame created from a dictionary:\n{df}")
print("-" * 50)

# 3. Create and display another dataframe with a custom index
df_indexed = pd.DataFrame(data,index=['Emp1','Emp2','Emp3','Emp4','Emp5'])
print("---- 2. Create a Data Frame with a custom index from a Dictionary ----")
print(f"Employee Details indexed dataframe:\n{df_indexed}")
print("-" * 50)

# 4. Dataframe attributes and information
print("---- 3. Dataframe attributes and information ----")
print(f"First 3 rows of employee details using 'head()' function:\n{df.head(3)}")
print(f"Last 2 rows of employee details using 'tail()' function:\n{df.tail(2)}")
print("-" * 50)

# 5. Accessing data from a Dataframe
print("---- 4. Accessing data from a Dataframe ----")
print(f"Single column (Series):\n{df['Name']}")
print(f"Multiple columns :\n{df[['Name','Age','Deparment']]}")
print(f"\nAccess by index location/position:\nFirst row:\n{df.iloc[0]}"
      f"\nSpecific cell (row 2, column 'Age'):\n{df.iloc[1,1]}")
print(f"Access by label (loc):\nEmployee with index 2:\n{df.loc[2]}")
print("-" * 50)

# 6. Create a Dataframe from a list of lists
print("---- 5. Create a Dataframe from a list of lists ----")
product_data = [
    ['Laptop', 99999.5, 'Electronics', 50],
    ['Mouse', 250.0, 'Electronics',200],
    ['Notebook',599.0, 'Stationary', 150],
    ['Pen',199.0,'Stationary',500]
]
product_columns = ['Product','Price','Category','Stock']
product_df = pd.DataFrame(product_data, columns=product_columns)
print(f"Product DataFrame:\n{product_df}")
print("-" * 50)

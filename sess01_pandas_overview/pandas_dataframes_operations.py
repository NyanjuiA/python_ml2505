# Python script to demonstrate various operations on Pandas dataframes
# like filtering, grouping, merging, and other dataframe operations.

# Import the required module
import pandas as pd

# 1. Create a sample employee dataframe
print("---- 1. Create a sample employee dataframe ----")
employees = pd.DataFrame({
   "EmployeeID": [1001, 1002, 1003, 1004, 1005],
   'Name': ['Abigail', "Kamau", "Sharlene", 'Diana', "Mueni"],
   'Salary': [55000, 65000, 72000, 48000, 60000],
   'Department': ['HR', 'IT', "IT", 'Marketing', "Finance"]
})

# Create a sales dataframe
sales = pd.DataFrame({
   "EmployeeID": [1001, 1002, 1003, 1004, 1005],
   "Q1_Sales": [15000, 22000, 18000, 12000, 25000],
   "Q2_Sales": [18000, 24000, 21000, 14000, 28000],
   "Q3_Sales": [22000, 25000, 22500, 15500, 32000],
})

# Display the employee and sales dataframes
print("Employee Dataframe".center(55, "-"))
print(employees)
print()
print("Sales Dataframe".center(55, "-"))
print(sales)
print()

# 2. Merge the employee and sales dataframes (on the 'EmployeeID' field)
print("---- 2. Merge the employee and sales dataframes (on the 'EmployeeID' field) ----")
combined = pd.merge(employees, sales, on='EmployeeID')
print("Merged employee & sales dataframe".center(55, "-"))
print(combined)
print()

# 3. Adding calculated columns
print("---- 3. Adding calculated columns ----")
combined['Total_Sales'] = combined['Q1_Sales'] + combined['Q2_Sales'] + combined['Q3_Sales']
combined['Avg_Sales'] = combined['Total_Sales'] / 3.0
combined['Bonus'] = combined['Total_Sales'] * .02
print(f"Dataframe with calculated columns:\n{combined}")
print()

# 4. Filtering Employee data
print("---- 4. Filtering Employee data ----")
print("Employees in the IT Department:")
it_employees = employees.loc[employees['Department'] == 'IT']
print(it_employees)
print("\nEmployees earning more than Kes. 60000")
high_salary = employees.loc[employees['Salary'] > 60000]
print(high_salary)
print()

# 5. Grouping and Aggregating data
print("---- 5. Grouping and Aggregating data ----")
department_stats = combined.groupby(['Department']).agg({
   'Salary':['mean','sum','min','max'],
   'Total_Sales':['mean','sum'],
   'EmployeeID':'count'
}).round(2)
print('Department Statistics:')
print(department_stats)
print()

# 6. Sorting data
print("---- 6. Sorting data ----")
print("Sorted by Total Sales (descending):")
sorted_by_sales = combined.sort_values(by='Total_Sales', ascending=False)
print(sorted_by_sales[['Name',"Department",'Total_Sales','Salary']])
print()

# 7. Statistical Summary of data
print("---- 7. Statistical Summary of data ----")
print(f"Summary statistics for combined dataframe:\n{combined.describe()}")


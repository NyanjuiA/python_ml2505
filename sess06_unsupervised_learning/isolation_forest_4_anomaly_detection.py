# Python file to demonstrate anomaly detection using Isolation Forest

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Generate a synthetic dataset for income and spending score
np.random.seed(42)

# Create a normal data point around income ~(30k - 70k) and Spending score ~(20 - 80)
income = np.random.normal(50_000,10_000, 300)
spending_score = np.random.normal(50,15, 300)
X = np.column_stack((income, spending_score))

# Introduce the anomalies with higher or lower income and spending scores
anomalies = np.array([[100_000,10],[20000,90],[80_000,75],[35000,5],[60000,95]])
X = np.vstack((X,anomalies))

# Convert the data into a pandas dataframe
df = pd.DataFrame(X, columns = ['Income', 'Spending_score'])

# Display the first five records
print(f"The first five rows are:\n{df.head(5)}")

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=.02, random_state=42)
df['Anomaly Score'] = iso_forest.fit_predict(df[['Income', 'Spending_score']])

# Separate normal points and anomalies for visualisation
normal = df[df['Anomaly Score'] == 1]
anomaly = df[df['Anomaly Score'] == -1]

# Plot the results
plt.figure(figsize = (10,8))
plt.scatter(normal['Income'],normal['Spending_score'],
            color='blue', label='Normal',alpha=.6)
plt.scatter(anomaly['Income'],anomaly['Spending_score'],
            color='red', label='Anomaly',marker='x',s=100)
plt.xlabel('Income')
plt.ylabel('Spending_Score')
plt.title('Anomaly Detection with Isolation Forest')
plt.legend()
plt.grid(True)
plt.show()
# Python file to demonstrate Heirachical Clustering Algorithm

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Generate a synthetic customer dataset with income and spending score
data = {
   'Income' : np.random.randint(20000,100000,100),
   'Spending Score':np.random.randint(1,50,100)
}

# Create a dataframe and display the first 10 rows
df = pd.DataFrame(data)
print(f"The frist 10 rows of the dataframe are:\n{df.head(10)}")

# Visualise the entire dataset
plt.figure(figsize=(10,8))
plt.scatter(df['Income'], df['Spending Score'])
plt.title("Synthetic Dataset: Income vs Spending Score")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

# Create a dendrogram
plt.figure(figsize=(10,8))
dendogram = sch.dendrogram(sch.linkage(df, method='ward'))
plt.title("Dendogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distances")
plt.show()

# Apply Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')
df['Cluster'] = hc.fit_predict(df)

# Plot the Clusters
plt.figure(figsize=(10,8))
plt.scatter(df['Income'], df['Spending Score'],c=df['Cluster'],cmap='viridis')
plt.title("Heirachical Clustering: Income vs Spending Score")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()
# Python file to demonstrate applying t-Stochastic Neighbour Embeddig on a
# synthetic dataset to visualise customer clusters based on their financial behaviour

# Import the required behaviour
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

# Variables to be used in generating a synthetic dataset
n_samples = 300
n_features = 4 # Number of customer features e.g age, income, gender, spending score & savings
centers = 5 # Number of customer clusters
cluster_std= 1.5 # Standard deviation of the clusters

# Create the customer features
features = ['Age','Income','Spending Score', 'Savings']

# Use make_blobs to generate a synthetic dataset
X,y = make_blobs(n_samples=n_samples,n_features=n_features,centers=centers,
                 cluster_std=cluster_std, random_state=42)

# Apply tSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(10,8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.legend(scatter.legend_elements(),title="Classes")
plt.title('t-SNE Visualisation of Synthetic Customer Data')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.colorbar(scatter)
plt.show()

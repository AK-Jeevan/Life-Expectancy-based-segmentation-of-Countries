# This Model will Groups countries based on their life expectancy and other health-related metrics and helps to identifies Outliers in the data.
# We will be using DBSCAN Clustering.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\akjee\Documents\AI\ML\Unsupervised Learning\life_expectancy.csv")

# Step 2: Remove duplicates
data.drop_duplicates(inplace=True)

# Step 3: Separate country names and keep only numeric columns
countries = data['Country']
X = data.drop(columns=['Country'])
X = X.select_dtypes(include=[np.number])  # Keep only numeric features
X = X.dropna()  # Remove rows with missing values
countries = countries.loc[X.index]  # Align countries with cleaned data

# Step 4: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Apply DBSCAN clustering
dbscan = DBSCAN(eps=1.0, min_samples=5)  # You can change these values
labels = dbscan.fit_predict(X_scaled)

# Step 7: Show cluster labels
result = pd.DataFrame({
    'Country': countries.values,
    'Cluster': labels
})
print(result.head())

# Step 8: Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=50)
plt.title('DBSCAN Clusters (Scaled Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Step 9: Print outliers
outliers = result[result['Cluster'] == -1]
print(f"\nNumber of outliers: {len(outliers)}")
print("Outlier countries:")
print(outliers['Country'].tolist())

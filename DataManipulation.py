import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and clean the data
data = pd.read_csv('customer_orders.csv', parse_dates=['order_date'])

# Check for missing values and handle them
data.isnull().sum()
data.dropna(inplace=True)

# Step 2: Feature engineering
# Calculate total spend, number of orders, and average order value for each customer
customer_summary = data.groupby('customer_id').agg({
    'order_id': pd.Series.nunique,
    'quantity': 'sum',
    'price': 'sum'
}).reset_index()

customer_summary.columns = ['customer_id', 'num_orders', 'total_quantity', 'total_spend']

# Calculate average order value
customer_summary['avg_order_value'] = customer_summary['total_spend'] / customer_summary['num_orders']

# Step 3: Perform customer segmentation using K-Means clustering
# Standardize the features
features = customer_summary[['num_orders', 'total_quantity', 'total_spend', 'avg_order_value']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the elbow plot, let's choose 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
customer_summary['cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Analyze and visualize the clusters
# Calculate mean values of features for each cluster
cluster_analysis = customer_summary.groupby('cluster').agg({
    'num_orders': 'mean',
    'total_quantity': 'mean',
    'total_spend': 'mean',
    'avg_order_value': 'mean'
}).reset_index()

print(cluster_analysis)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_summary, x='total_spend', y='avg_order_value', hue='cluster', palette='viridis', s=100)
plt.title('Customer Segmentation based on Total Spend and Average Order Value')
plt.xlabel('Total Spend')
plt.ylabel('Average Order Value')
plt.legend(title='Cluster')
plt.show()

# Save the clustered data
customer_summary.to_csv('customer_segmentation.csv', index=False)

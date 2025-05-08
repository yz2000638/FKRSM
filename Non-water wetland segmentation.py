import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
file_path = r"C:\Users...."
NDVI_data = pd.read_csv(file_path)

# Data cleaning and preprocessing
NDVI_data['NDVI Count'] = NDVI_data['NDVI Count'].replace(',', '', regex=True).astype(float)

# Extract NDVI Band Value data
band_values = NDVI_data['Band Value'].values

# Define number of clusters (e.g., 4 if city has 3 types of non-water wetland, 3 if only 2 types)
n_clusters = 4  # Modify this parameter to define the number of classes

# Apply KMeans clustering to divide Band Value into n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
NDVI_data['Cluster'] = kmeans.fit_predict(band_values.reshape(-1, 1))

# Get and sort cluster centers to determine thresholds
centers = sorted(kmeans.cluster_centers_.flatten())

# Calculate midpoint between each pair of adjacent cluster centers as threshold
thresholds = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

# Define color list for visualizing threshold lines
colors = ['red', 'green', 'blue', 'purple', 'orange']  # Add more colors for additional clusters if needed

# Plot NDVI histogram with threshold lines
plt.figure(figsize=(12, 6))
plt.plot(band_values, NDVI_data['NDVI Count'], label='NDVI Count')
for i, threshold in enumerate(thresholds):
    plt.axvline(x=threshold, color=colors[i % len(colors)], linestyle='--', label=f'Threshold {i+1} ({threshold:.3f})')

# Add chart title and labels
plt.title('NDVI Count with K-means Thresholds')
plt.xlabel('NDVI')
plt.ylabel('Count')
plt.legend()

# Save figure locally with high resolution
save_path = r"C:\Users...."
plt.savefig(save_path, dpi=300)  # Save the figure with 300 DPI for high quality

plt.show()

# Output the computed threshold values
print("Thresholds:", thresholds)

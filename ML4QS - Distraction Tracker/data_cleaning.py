import pandas as pd
import matplotlib.pyplot as plt
from outlier_detection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection

file_path = "./data-3-prepped/merged_data.csv"

data_table = pd.read_csv(file_path)

distribution_based_outlier_detection = DistributionBasedOutlierDetection()
distance_based_outlier_detection = DistanceBasedOutlierDetection()

for col in data_table.columns:
    if col != "Time (s)":
        data_table = distribution_based_outlier_detection.chauvenet(data_table, col, C=3)
        # data_table = distribution_based_outlier_detection.mixture_model(data_table, col)

cols_for_distance_based_detection = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
# data_table = distance_based_outlier_detection.simple_distance_based(data_table, cols_for_distance_based_detection,
#                                                                     d_function='euclidean', dmin=1, fmin=0.1)

print(data_table.head())

for col in data_table.columns:
    if '_outlier' in col:
        # Get the original column name
        original_col = col.replace('_outlier', '')
        
        # Plot the original column values
        plt.figure(figsize=(10, 6))
        plt.plot(data_table['Time (s)'], data_table[original_col], label=original_col, color='blue')
        
        # Highlight the outliers in red
        outliers = data_table[data_table[col]]
        plt.scatter(outliers['Time (s)'], outliers[original_col], color='red', label='Outliers')
        
        plt.xlabel('Time (s)')
        plt.ylabel(original_col)
        plt.title(f'{original_col} with Outliers Highlighted')
        plt.legend()
        plt.show()

data_table.to_csv("./data-3-prepped/outlier_detection_results.csv", index=False)
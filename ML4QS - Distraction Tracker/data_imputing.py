import pandas as pd
import numpy as np

file_path = "./data-4-prepped/merged_data.csv"

data = pd.read_csv(file_path, index_col=0)


# Interpolate the dataset based on previous/next values..
def impute_interpolate(dataset, col):
    dataset[col] = dataset[col].interpolate()
    # And fill the initial data points if needed:
    dataset[col] = dataset[col].fillna(method='bfill')
    return dataset


numeric_features = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                    'Absolute acceleration (m/s^2)', 'Attitude x (m/s^2)', 'Attitude y (m/s^2)',
                    'Attitude z (m/s^2)', 'Absolute Attitude (m/s^2)', 'Tilt up/down (deg)_x',
                    'Tilt left/right (deg)_x', 'Gyroscope x (rad/s)', 'Gyroscope y (rad/s)',
                    'Gyroscope z (rad/s)', 'Absolute (rad/s)', 'Linear Acceleration x (m/s^2)',
                    'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)',
                    'Absolute Linear acceleration (m/s^2)', 'Inclination (deg)', 'Rotation (deg)',
                    'Tilt up/down (deg)_y', 'Tilt left/right (deg)_y', 'Tilt up/down (deg)',
                    'Tilt left/right (deg)', 'Sound pressure level (dB)']
eye_features = ['Left Pupil X', 'Left Pupil Y', 'Right Pupil X', 'Right Pupil Y',
                'Blinking', 'Looking Center', 'Looking Left', 'Looking Right']
mouse_features = ['Event', 'PositionX', 'PositionY']

missing_values = data.isnull().sum()

print("Missing values in each feature:")
print(missing_values)

data[eye_features] = data[eye_features].fillna(-1)

data[mouse_features] = data[mouse_features].fillna(0)

for col in numeric_features:
    data = impute_interpolate(data, col)

data.to_csv('./data-4-prepped/data_imputed.csv')

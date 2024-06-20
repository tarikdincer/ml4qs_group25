import pandas as pd
from datetime import datetime
import os
import math

base_dir = "./data-4/"
output_dir = "./data-4-prepped/"

def create_output_directory():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_starting_time():
    data = pd.read_csv(os.path.join(base_dir, 'meta/time.csv'))
    start_row = data[data["event"] == "START"]

    start_time_text = start_row["system time text"].values[0]
    start_datetime = datetime.strptime(start_time_text, "%Y-%m-%d %H:%M:%S.%f UTC%z")

    return start_datetime.replace(tzinfo=None)

def seperate_raw_data():
    column_names = [
        "Time_Acc", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)",
        "Time_LinAcc", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)", "Absolute Linear acceleration (m/s^2)",
        "Time_Att", "Attitude x (m/s^2)", "Attitude y (m/s^2)", "Attitude z (m/s^2)", "Absolute Attitude (m/s^2)",
        "Time_Gyro", "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)", "Absolute (rad/s)"
    ]

    input_file = os.path.join(base_dir, 'Raw Data.csv')
    data = pd.read_csv(input_file, names=column_names, header=0)

    gyroscope_data = data[["Time_Gyro", "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)", "Absolute (rad/s)"]]
    gyroscope_data.rename(columns={"Time_Gyro": "Time (s)"}, inplace=True)
    gyroscope_data.to_csv(os.path.join(output_dir, 'gyroscope_data.csv'), index=False)

    attitude_data = data[["Time_Att", "Attitude x (m/s^2)", "Attitude y (m/s^2)", "Attitude z (m/s^2)", "Absolute Attitude (m/s^2)"]]
    attitude_data.rename(columns={"Time_Att": "Time (s)"}, inplace=True)
    attitude_data.to_csv(os.path.join(output_dir, 'attitude_data.csv'), index=False)

    acceleration_data = data[["Time_Acc", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]]
    acceleration_data.rename(columns={"Time_Acc": "Time (s)"}, inplace=True)
    acceleration_data.to_csv(os.path.join(output_dir, 'acceleration_data.csv'), index=False)

    linear_acceleration_data = data[["Time_LinAcc", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)", "Absolute Linear acceleration (m/s^2)"]]
    linear_acceleration_data.rename(columns={"Time_LinAcc": "Time (s)"}, inplace=True)
    linear_acceleration_data.to_csv(os.path.join(output_dir, 'linear_acceleration_data.csv'), index=False)

def prep_amplitude():
    sound_pressure_columns = ["Time (s)", "Sound pressure level (dB)"]

    input_file = os.path.join(base_dir, 'Amplitudes.csv')
    sound_data = pd.read_csv(input_file, names=sound_pressure_columns, header=0)

    sound_data['Sound pressure level (dB)'] = pd.to_numeric(sound_data['Sound pressure level (dB)'], errors='coerce')

    sound_data = sound_data[pd.notnull(sound_data['Sound pressure level (dB)'])]

    sound_data.to_csv(os.path.join(output_dir, 'amplitudes_data.csv'), index=False)

def prep_flat():
    tilt_columns = ["Time (s)", "Tilt up/down (deg)", "Tilt left/right (deg)"]

    tilt_input_file = os.path.join(base_dir, 'Flat.csv')
    tilt_data = pd.read_csv(tilt_input_file, names=tilt_columns, header=0)

    tilt_data.to_csv(os.path.join(output_dir, 'flat_data.csv'), index=False)

def prep_plane():
    plane_columns = ["Time (s)", "Inclination (deg)", "Rotation (deg)"]

    plane_input_file = os.path.join(base_dir, 'Plane.csv')
    plane_data = pd.read_csv(plane_input_file, names=plane_columns, header=0)

    plane_data.to_csv(os.path.join(output_dir, 'plane_data.csv'), index=False)

def prep_side():
    side_columns = ["Time (s)", "Tilt up/down (deg)", "Tilt left/right (deg)"]

    side_input_file = os.path.join(base_dir, 'Side.csv')
    side_data = pd.read_csv(side_input_file, names=side_columns, header=0)

    side_data.to_csv(os.path.join(output_dir, 'side_data.csv'), index=False)

def prep_upright():
    upright_columns = ["Time (s)", "Tilt up/down (deg)", "Tilt left/right (deg)"]

    upright_input_file = os.path.join(base_dir, 'Upright.csv')
    upright_data = pd.read_csv(upright_input_file, names=upright_columns, header=0)

    upright_data.to_csv(os.path.join(output_dir, 'upright_data.csv'), index=False)

def prep_gaze_tracking():
    pupil_columns = ["Timestamp", "Left Pupil X", "Left Pupil Y", "Right Pupil X", "Right Pupil Y", "Blinking", "Looking Center", "Looking Left", "Looking Right"]

    pupil_input_file = os.path.join(base_dir, 'gaze_tracking_log.csv')
    pupil_data = pd.read_csv(pupil_input_file, names=pupil_columns, header=0)

    start_time = get_starting_time()
    pupil_data["Time (s)"] = pupil_data["Timestamp"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S,%f") - start_time).total_seconds())
    pupil_data = pupil_data[pupil_data["Time (s)"] >= 0]

    pupil_data = pupil_data[["Time (s)", "Left Pupil X", "Left Pupil Y", "Right Pupil X", "Right Pupil Y", "Blinking", "Looking Center", "Looking Left", "Looking Right"]]

    pupil_data.to_csv(os.path.join(output_dir, 'gaze_tracking_data.csv'), index=False)

def prep_key_mouse():
    event_columns = ["Time", "Event", "Key/Button", "PositionX", "PositionY"]

    event_input_file = os.path.join(base_dir, 'log.csv')
    with open(event_input_file, 'r') as file:
        lines = file.readlines()

    cleaned_lines = [line.replace(",,,", ",,") for line in lines]

    temp_file = os.path.join(base_dir, 'temp_log.csv')
    with open(temp_file, 'w') as file:
        file.writelines(cleaned_lines)
    
    event_data = pd.read_csv(temp_file, names=event_columns, header=0)

    start_time = get_starting_time()
    event_data["Time (s)"] = event_data["Time"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") - start_time).total_seconds())
    event_data = event_data[event_data["Time (s)"] >= 0]

    event_data = event_data[["Time (s)", "Event", "Key/Button", "PositionX", "PositionY"]]

    event_data.to_csv(os.path.join(output_dir, 'mouse_key_data.csv'), index=False)

    distractions = []
    recording = False
    start_time = None

    for index, row in event_data.iterrows():
        if row["Key/Button"] == "`":
            if recording:
                end_time = row["Time (s)"]
                distractions.extend(list(range(int(start_time), int(end_time) + 1)))
                recording = False
            else:
                start_time = row["Time (s)"]
                recording = True

    distractions = sorted(set(distractions))

    distractions_df = pd.DataFrame(distractions, columns=["Time (s)"])
    distractions_output_file = os.path.join(output_dir, 'distractions.csv')
    distractions_df.to_csv(distractions_output_file, index=False)

def prep_data_files():
    create_output_directory()
    seperate_raw_data()
    prep_amplitude()
    prep_flat()
    prep_plane()
    prep_side()
    prep_upright()
    prep_gaze_tracking()
    prep_key_mouse()

def resample_phone_sensors(input_file, output_file):
    data = pd.read_csv(input_file)

    data['Time (s)'] = pd.to_datetime(data['Time (s)'], unit='s')

    start_time = data['Time (s)'].min()
    data['Time (s)'] = (data['Time (s)'] - start_time).dt.total_seconds()

    data['Time (s)'] = data['Time (s)'].round(0)

    resampled_data = data.groupby('Time (s)').mean().reset_index()

    resampled_data.to_csv(output_file, index=False)

def resample_mouse_key(input_file, output_file):
    data = pd.read_csv(input_file)

    data['Time (s)'] = pd.to_datetime(data['Time (s)'], unit='s')

    data['Elapsed Time (s)'] = (data['Time (s)'] - data['Time (s)'].min()).dt.total_seconds()

    data['Elapsed Time (s)'] = data['Elapsed Time (s)'].apply(lambda x: math.floor(x))
    
    d = pd.get_dummies(data['Event'])
    data = pd.concat([data, d], axis=1)
    print(data)
    data.rename(columns=lambda x: x.strip(), inplace=True)

    grouped_data = data.groupby('Elapsed Time (s)').agg({
        'Mouse Move': 'sum',
        'Key Press': 'sum',
        'Mouse Scroll': 'sum',
        'Mouse Click': 'sum',
        'PositionX': 'mean',
        'PositionY': 'mean'
    }).reset_index()

    print(grouped_data)


    grouped_data.rename(columns={'Elapsed Time (s)': 'Time (s)'}, inplace=True)
    grouped_data.to_csv(output_file, index=False)


def resample_gaze_tracking(input_file, output_file):
    data = pd.read_csv(input_file)

    data['Time (s)'] = pd.to_datetime(data['Time (s)'], unit='s')

    data['Elapsed Time (s)'] = (data['Time (s)'] - data['Time (s)'].min()).dt.total_seconds()

    data['Elapsed Time (s)'] = data['Elapsed Time (s)'].apply(lambda x: math.floor(x))

    grouped_data = data.groupby('Elapsed Time (s)').agg({
        "Left Pupil X": 'mean',
        "Left Pupil Y": 'mean',
        "Right Pupil X": 'mean',
        "Right Pupil Y": 'mean',
        "Blinking": 'sum',
        "Looking Center": 'sum',
        "Looking Left": 'sum',
        "Looking Right": 'sum'
    }).reset_index()

    grouped_data.rename(columns={'Elapsed Time (s)': 'Time (s)'}, inplace=True)
    grouped_data.to_csv(output_file, index=False)

def sample_data():
    resample_phone_sensors(os.path.join(output_dir, 'acceleration_data.csv'), os.path.join(output_dir, 'acceleration_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'amplitudes_data.csv'), os.path.join(output_dir, 'amplitudes_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'attitude_data.csv'), os.path.join(output_dir, 'attitude_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'flat_data.csv'), os.path.join(output_dir, 'flat_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'plane_data.csv'), os.path.join(output_dir, 'plane_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'side_data.csv'), os.path.join(output_dir, 'side_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'upright_data.csv'), os.path.join(output_dir, 'upright_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'gyroscope_data.csv'), os.path.join(output_dir, 'gyroscope_data.csv'))
    resample_phone_sensors(os.path.join(output_dir, 'linear_acceleration_data.csv'), os.path.join(output_dir, 'linear_acceleration_data.csv'))
    resample_mouse_key(os.path.join(output_dir, 'mouse_key_data.csv'), os.path.join(output_dir, 'mouse_key_data.csv'))
    resample_gaze_tracking(os.path.join(output_dir, 'gaze_tracking_data.csv'), os.path.join(output_dir, 'gaze_tracking_data.csv'))

def merge_data_files(file_names, distractions_file):
    merged_df = pd.DataFrame()
    
    for file_name in file_names:
        file_path = os.path.join(output_dir, file_name)
        df = pd.read_csv(file_path)
        
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Time (s)', how='outer')
    
    distractions_df = pd.read_csv(os.path.join(output_dir, distractions_file))
    
    merged_df['isDistracted'] = merged_df['Time (s)'].apply(lambda x: 1 if x in distractions_df['Time (s)'].values else 0)
    
    merged_df.to_csv(os.path.join(output_dir, 'merged_data.csv'), index=False)

if __name__ == "__main__":
    prep_data_files()
    sample_data()

    file_names = [
    'acceleration_data.csv',
    'amplitudes_data.csv',
    'attitude_data.csv',
    'flat_data.csv',
    'gaze_tracking_data.csv',
    'gyroscope_data.csv',
    'linear_acceleration_data.csv',
    'mouse_key_data.csv',
    'plane_data.csv',
    'side_data.csv',
    'upright_data.csv',
    ]
    distractions_file = 'distractions.csv'

    merge_data_files(file_names, distractions_file)

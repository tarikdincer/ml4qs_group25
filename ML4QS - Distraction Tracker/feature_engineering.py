import pandas as pd
from temporal_abstraction import NumericalAbstraction, CategoricalAbstraction

file_path = "./data-4-prepped/data_imputed.csv"

df = pd.read_csv(file_path, index_col=0)

numerical_abstraction = NumericalAbstraction()
categorical_abstraction = CategoricalAbstraction()

cols_for_abstraction = ['Absolute acceleration (m/s^2)', 'Sound pressure level (dB)', 'Absolute Attitude (m/s^2)', 'Absolute (rad/s)', 'Absolute Linear acceleration (m/s^2)', 'Inclination (deg)', 'Rotation (deg)']

df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=5, aggregation_function_name='mean')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=60, aggregation_function_name='mean')

df.to_csv("./data-4-prepped/aggregated_data.csv")

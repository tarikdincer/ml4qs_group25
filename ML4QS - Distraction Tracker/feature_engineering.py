import pandas as pd
from temporal_abstraction import NumericalAbstraction, CategoricalAbstraction
from pca import ReduceDimensionality

file_path = "./data-4-prepped/data_imputed.csv"

df = pd.read_csv(file_path, index_col=0)

numerical_abstraction = NumericalAbstraction()
categorical_abstraction = CategoricalAbstraction()

cols_for_abstraction = ['Absolute acceleration (m/s^2)', 'Sound pressure level (dB)', 'Absolute Attitude (m/s^2)', 'Absolute (rad/s)', 'Absolute Linear acceleration (m/s^2)', 'Inclination (deg)', 'Rotation (deg)']

df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=5, aggregation_function_name='mean')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=60, aggregation_function_name='mean')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=5, aggregation_function_name='max')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=60, aggregation_function_name='max')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=5, aggregation_function_name='min')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=60, aggregation_function_name='min')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=5, aggregation_function_name='std')
df = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=60, aggregation_function_name='std')
df.fillna(-1, inplace=True)

print(df)

# categorical_features = ['isDistracted', 'Blinking', 'Looking Center', 'Looking Left', 'Looking Right', 'Event']
# numerical_features = [feature for feature in df.columns if feature not in categorical_features]
# print(len(numerical_features))
# pca = ReduceDimensionality(df, numerical_features)
# pca.seperate_numerical()
# pca.standardize_data()
# pca.find_optimal_num_compontents()
# pca.create_graph()
# df = pca.create_final_data()

print(df)


df.to_csv("./data-4-prepped/aggregated_data.csv")

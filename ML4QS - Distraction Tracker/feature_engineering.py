import pandas as pd
from temporal_abstraction import NumericalAbstraction, CategoricalAbstraction

file_path = "./data-3-prepped/data_imputed.csv"

df = pd.read_csv(file_path, index_col=0)

numerical_abstraction = NumericalAbstraction()
categorical_abstraction = CategoricalAbstraction()

cols_for_abstraction = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

aggregated_data = numerical_abstraction.abstract_numerical(data_table=df, cols=cols_for_abstraction, window_size=3, aggregation_function_name='mean')

df.to_csv("./data-3-prepped/aggregated_data.csv")

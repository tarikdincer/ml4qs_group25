import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Eda:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data_table = None

    def load_data(self):
        self.data_table = pd.read_csv(self.file_path)

    def print_stats(self, cols):
        print(len(self.data_table.columns))
        print(self.data_table.columns)
        print(self.data_table[cols].describe())

    def plot_column_distribution_over_time(self, column_name):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data_table['Time (s)'], self.data_table[column_name], label=column_name)
        plt.xlabel('Time (s)')
        plt.ylabel(column_name)
        plt.title(f'Distribution of {column_name} over Time')
        plt.legend()
        plt.show()

    def print_timesteps_count(self):
        timesteps_count = len(self.data_table)
        print(f"Total number of timesteps: {timesteps_count}")

    def print_unique_values_count(self, column_name):
        unique_values_count = self.data_table[column_name].nunique()
        print(f"Total number of unique values in column '{column_name}': {unique_values_count}")

    def print_isdistracted_count(self):
        is_distracted_count = self.data_table['isDistracted'].sum()
        print(f"Total number of timesteps labeled as distracted: {is_distracted_count}")

    def distracted_sessions_stats(self):
        distracted_indices = self.data_table[self.data_table['isDistracted'] == 1].index
        distracted_sessions_lengths = []
        start_index = distracted_indices[0]
        last_index = distracted_indices[0]

        for index in distracted_indices[1:]:
            if index == last_index + 1:
                last_index = index
                continue
            else:
                distracted_sessions_lengths.append(last_index - start_index)
                start_index = index
                last_index = index

        print(f"Total number of distracted sessions: {len(distracted_sessions_lengths)}")
        print(f"Duration of each distracted session (in timesteps): {distracted_sessions_lengths}")
        print(f"Average duration of distracted session: {sum(distracted_sessions_lengths) / len(distracted_sessions_lengths)}")

    def plot_dataset(self, columns, match='exact', display='line'):
        names = list(self.data_table.columns)

        # Create subplots if more columns are specified.
        if len(columns) > 1:
            f, xar = plt.subplots(len(columns), sharex=True, sharey=False)
        else:
            f, xar = plt.subplots()
            xar = [xar]

        f.subplots_adjust(hspace=0.4)

        # Pass through the columns specified.
        for i in range(0, len(columns)):
            xar[i].set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

            # Select the relevant column
            if match == 'exact':
                relevant_col = columns[i]
            elif match == 'like':
                relevant_col = [name for name in names if columns[i] in name]
                if not relevant_col:
                    raise ValueError(f"No columns match the pattern {columns[i]}")
                relevant_col = relevant_col[0]
            else:
                raise ValueError("Match should be 'exact' or 'like'.")

            max_value = self.data_table[relevant_col].replace([np.inf, -np.inf], np.nan).max()
            min_value = self.data_table[relevant_col].replace([np.inf, -np.inf], np.nan).min()

            # Create a mask to ignore the NaN and Inf values when plotting:
            mask = self.data_table[relevant_col].replace([np.inf, -np.inf], np.nan).notnull()

            # Display point, or as a line
            if display == 'points':
                xar[i].plot(self.data_table['Time (s)'][mask], self.data_table[relevant_col][mask], '+')
            else:
                xar[i].plot(self.data_table['Time (s)'][mask], self.data_table[relevant_col][mask], '-')

            xar[i].tick_params(axis='y', labelsize=10)
            xar[i].legend([relevant_col], fontsize='xx-small', numpoints=1, loc='upper center',
                        bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True)

            xar[i].set_ylim([min_value - 0.1 * (max_value - min_value),
                            max_value + 0.1 * (max_value - min_value)])

        # Make sure we get a nice figure with only a single x-axis and labels there.
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel('Time (s)')
        # plt.savefig("plot_session1.png")
        plt.show()

if __name__ == "__main__":
    eda = Eda("./data-3-prepped/merged_data.csv")
    eda.load_data()
    
    # column_name = 'Acceleration x (m/s^2)'
    # eda.plot_column_distribution_over_time(column_name)

    eda.print_timesteps_count()

    column_name = 'Acceleration y (m/s^2)'
    eda.print_unique_values_count(column_name)

    eda.print_isdistracted_count()
    eda.distracted_sessions_stats()

    relevant_columns = ['Sound pressure level (dB)', 'Absolute acceleration (m/s^2)', 'Rotation (deg)', 'Inclination (deg)']
    eda.print_stats(relevant_columns)
    eda.plot_dataset(relevant_columns, match='exact')

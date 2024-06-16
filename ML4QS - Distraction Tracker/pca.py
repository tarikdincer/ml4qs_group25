# do pca on the entire numerical data set
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class ReduceDimensionality:

    data = None
    numerical_data = None
    categorical_data = None
    optimal_num_components = None
    numerical_features = None
    scaled_numerical_data = None
    var_ratios = None
    pca_df = None
    final_df = None
    
    def __init__(self, data, numerical_features):
        self.data = data
        self.numerical_features = numerical_features

    
    def seperate_numerical(self):
        self.numerical_data = self.data[self.numerical_features]
        categorical_columns = [col for col in self.data.columns if col not in self.numerical_features]
        self.categorical_data = self.data[categorical_columns]

    
    def standardize_data(self):
        std_scaler = StandardScaler()
        self.scaled_numerical_data = std_scaler.fit_transform(self.numerical_data, )
        self.scaled_numerical_data = pd.DataFrame(self.scaled_numerical_data, columns=self.numerical_data.columns)
        print(self.scaled_numerical_data)

    
    def find_optimal_num_compontents(self):
        # list the sum of the explained variance for each amount of components.
        # optimal number is at least 90% of the variance explained
        self.var_ratios = []
        num_features = len(self.scaled_numerical_data.columns)
        for i in range(1, num_features + 1):
            pca = PCA(n_components=i)
            pca.fit(self.scaled_numerical_data)
            self.var_ratios.append(np.sum(pca.explained_variance_ratio_))
        
        for index, ratio in enumerate(self.var_ratios):
            if ratio >= 0.90:
                self.optimal_num_components = index
                break

        print("optimal number of components found: " + str(self.optimal_num_components))
        

    def create_graph(self):
        num_features = len(self.scaled_numerical_data.columns)
        plt.figure(figsize=(4,2),dpi=150)
        plt.grid()
        plt.plot(num_features, self.var_ratios,marker='o')
        plt.xlabel('n_components')
        plt.ylabel('Explained variance ratio')
        plt.title('n_components vs. Explained Variance Ratio')

    
    def create_final_data(self):
        pca = PCA(n_components=self.optimal_num_components)
        self.pca_df = pca.fit_transform(self.scaled_numerical_data)
        self.final_df = pd.concat([self.pca_df, self.categorical_data])

        return self.final_df
        

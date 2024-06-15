import pandas as pd
import numpy as np
from prepare_dataset_for_learning import PrepareDatasetForLearning
from feature_selection import FeatureSelectionClassification
from learning_algorithms import ClassificationAlgorithms
from evaluation import ClassificationEvaluation

file_path = "./data-3-prepped/aggregated_data.csv"
prep_dataset = PrepareDatasetForLearning()
feature_selection = FeatureSelectionClassification()
classification_algorithms = ClassificationAlgorithms()
evaluation = ClassificationEvaluation()

df = pd.read_csv(file_path, index_col=None)
df.fillna(-1, inplace=True)
train_X, test_X, train_Y, test_Y = prep_dataset.split_single_dataset_classification(df, ['isDistracted'], matching='', training_frac = 0.8, filter=False, temporal=True, random_state = 42)

selected_features, ordered_features, ordered_scores = feature_selection.forward_selection(20, X_train=train_X, X_test=test_X, y_train=train_Y, y_test=test_Y, gridsearch=True)
train_X = train_X[selected_features]
test_X = test_X[selected_features]
pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = classification_algorithms.decision_tree(train_X, train_Y, test_X)

accuracy_score = evaluation.accuracy(test_Y, pred_test_y)
precision = evaluation.precision(test_Y, pred_test_y)
f1 = evaluation.f1(test_Y, pred_test_y)
print(accuracy_score)
print(precision)
print(f1)
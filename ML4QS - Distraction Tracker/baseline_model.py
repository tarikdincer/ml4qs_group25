import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score
import matplotlib.pyplot as plt
from prepare_dataset_for_learning import PrepareDatasetForLearning
from feature_selection import FeatureSelectionClassification
from learning_algorithms import ClassificationAlgorithms
from evaluation import ClassificationEvaluation

prep_dataset = PrepareDatasetForLearning()
feature_selection = FeatureSelectionClassification()
classification_algorithms = ClassificationAlgorithms()
evaluation = ClassificationEvaluation()

file_paths = ["./data-1-prepped/aggregated_data.csv", "./data-2-prepped/aggregated_data.csv",
              "./data-3-prepped/aggregated_data.csv", "./data-4-prepped/aggregated_data.csv"]
dfs = []

for fpath in file_paths:
    df = pd.read_csv(fpath, index_col=None)
    df.fillna(-1, inplace=True)
    dfs.append(df)

# Prepare datasets
train_X, test_X, train_Y, test_Y = prep_dataset.split_multiple_datasets_classification(dfs, ['isDistracted'], matching='', training_frac=0.8, filter=False, temporal=True, random_state=42)
print(f'Training set size: {train_X.shape[0]}')
print(f'Test set size: {test_X.shape[0]}')

# Feature selection
selected_features, ordered_features, ordered_scores = feature_selection.forward_selection(20, X_train=train_X, X_test=test_X, y_train=train_Y, y_test=test_Y, gridsearch=True)
train_X = train_X[selected_features]
test_X = test_X[selected_features]

def plot_confusion_matrix_and_metrics(model_name, conf_matrix):
    accuracy = accuracy_score(test_Y, pred_test_y)
    precision = precision_score(test_Y, pred_test_y)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Distracted', 'Not Distracted'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f"{model_name} Confusion Matrix")
    plt.show()
    
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print("")

# Decision Tree
pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = classification_algorithms.decision_tree(train_X, train_Y, test_X)
conf_matrix_dt = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("Decision Tree", conf_matrix_dt)

# Random Forest
pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = classification_algorithms.random_forest(train_X, train_Y, test_X)
conf_matrix_rf = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("Random Forest", conf_matrix_rf)

# SVM Evaluation
pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = classification_algorithms.support_vector_machine_with_kernel(train_X, train_Y, test_X)
conf_matrix_svm = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("SVM Evaluation", conf_matrix_svm)

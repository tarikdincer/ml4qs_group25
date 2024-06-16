import pandas as pd
import numpy as np
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score
import matplotlib.pyplot as plt
from prepare_dataset_for_learning import PrepareDatasetForLearning
from feature_selection import FeatureSelectionClassification
from learning_algorithms import ClassificationAlgorithms
from evaluation import ClassificationEvaluation
import optuna
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

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
selected_features, ordered_features, ordered_scores = feature_selection.forward_selection(20, X_train=train_X, X_test=test_X, y_train=train_Y, y_test=test_Y, gridsearch=False)
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

# Optuna objective functions for hyperparameter tuning

def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'accuracy',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'verbosity': -1
    }
    train_data = lgb.Dataset(train_X, label=train_Y)
    validation_data = lgb.Dataset(test_X, label=test_Y, reference=train_data)


    model = lgb.train(params, train_data, valid_sets=[validation_data])
    pred = model.predict(test_X, num_iteration=model.best_iteration)
    pred_labels = (pred > 0.5).astype(int)

    return accuracy_score(test_Y, pred_labels)
def objective_dt(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16)
    }
    model = DecisionTreeClassifier(**params)
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    return accuracy_score(test_Y, pred)

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16)
    }
    model = RandomForestClassifier(**params)
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    return accuracy_score(test_Y, pred)

def objective_svm(trial):
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e1, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 1e-1, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
    }
    model = SVC(**params)
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    return accuracy_score(test_Y, pred)

# Tuning LightGBM
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50)
best_params_lgb = study_lgb.best_params
best_accuracy_lgb = study_lgb.best_value
print("Best parameters for LightGBM: ", best_params_lgb)
print("Best accuracy for LightGBM: ", best_accuracy_lgb)

# Tuning Decision Tree
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective_dt, n_trials=50)
best_params_dt = study_dt.best_params
best_accuracy_dt = study_dt.best_value
print("Best parameters for Decision Tree: ", best_params_dt)
print("Best accuracy for Decision Tree: ", best_accuracy_dt)

# Tuning Random Forest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=50)
best_params_rf = study_rf.best_params
best_accuracy_rf = study_rf.best_value
print("Best parameters for Random Forest: ", best_params_rf)
print("Best accuracy for Random Forest: ", best_accuracy_rf)

# Tuning SVM
study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=50)
best_params_svm = study_svm.best_params
best_accuracy_svm = study_svm.best_value
print("Best parameters for SVM: ", best_params_svm)
print("Best accuracy for SVM: ", best_accuracy_svm)

# Training and evaluating models with the best parameters

#lightGBM
model_lgb = lgb.LGBMClassifier(**best_params_lgb)
model_lgb.fit(train_X, train_Y)
pred_test_y = model_lgb.predict(test_X)
conf_matrix_lgb = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("LightGBM", conf_matrix_lgb)

# Decision Tree
model_dt = DecisionTreeClassifier(**best_params_dt)
model_dt.fit(train_X, train_Y)
pred_test_y = model_dt.predict(test_X)
conf_matrix_dt = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("Decision Tree", conf_matrix_dt)

# Random Forest
model_rf = RandomForestClassifier(**best_params_rf)
model_rf.fit(train_X, train_Y)
pred_test_y = model_rf.predict(test_X)
conf_matrix_rf = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("Random Forest", conf_matrix_rf)

# SVM
model_svm = SVC(**best_params_svm)
model_svm.fit(train_X, train_Y)
pred_test_y = model_svm.predict(test_X)
conf_matrix_svm = evaluation.confusion_matrix(test_Y, pred_test_y, labels=[1, 0])
plot_confusion_matrix_and_metrics("SVM Evaluation", conf_matrix_svm)
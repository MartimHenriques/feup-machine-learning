from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,roc_curve, auc
from sklearn.model_selection import GridSearchCV, BaseCrossValidator, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy import ndarray
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

@dataclass
class Result:
    y_test: ndarray
    y_pred: ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: ndarray = None
    tpr: ndarray = None
    y_pred_proba: ndarray = None

    def toRow(self): 
        return [self.accuracy, self.precision, self.recall, self.f1]
    
    def plot_roc_curve(self):
        roc_auc = auc(self.fpr, self.tpr)
        if self.fpr is not None and self.tpr is not None and roc_auc is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()


'''Get metrics for the prediction'''
def predict(model, testing_inputs, testing_classes, df) -> Result:
    model.score(testing_inputs, testing_classes) #Why? I don't know

    fpr = None
    tpr = None
    if (hasattr(model, 'predict_proba')):
        y_pred_proba = model.predict_proba(testing_inputs)[:, 1]
        fpr, tpr, _ = roc_curve(testing_classes, y_pred_proba)

        df['pred'] = y_pred_proba
        threshold = { 
            'EA' : df[df['confID'] == 'EA']['pred'].nlargest(4).min(),
            'WE' : df[df['confID'] == 'WE']['pred'].nlargest(4).min()
            }

        y_pred = df.apply(lambda row: 1 if row['pred'] >= threshold[row['confID']] else 0, axis=1)
    else:
        y_pred = model.predict(testing_inputs)

    print(y_pred)

    accuracy = accuracy_score(testing_classes, y_pred)
    precision = precision_score(testing_classes, y_pred)
    recall = recall_score(testing_classes, y_pred)
    f1 = f1_score(testing_classes, y_pred)
    result = Result(testing_classes, y_pred, accuracy, precision, recall, f1,fpr, tpr)

    return result

'''Run a model and print results'''
def runModel(df, model, test_year=10):
    train_df = df[df['year'] < test_year]
    test_df = df[df['year'] == test_year]

    X_train = train_df.drop(columns=['playoff'] + (['confID'] if 'confID' in train_df.columns else []))
    y_train = train_df['playoff']

    X_test = test_df.drop(columns=['playoff'] + (['confID'] if 'confID' in test_df.columns else []))
    y_test = test_df['playoff']

    model.fit(X_train, y_train)
    return predict(model, X_test, y_test, test_df.copy())

def displayResults(result : Result):
    print(f"Accuracy: {round(result.accuracy * 100, 2)}%")
    print(f"Precision: {round(result.precision * 100, 2)}%")
    print(f"Recall: {round(result.recall * 100, 2)}%")
    print(f"F1-measure: {round(result.f1 * 100, 2)}%")

    cm = confusion_matrix(result.y_test, result.y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(5, 5))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def displayTree(df, test_year=10, max_depth=3):
    train_df = df[df['year'] < test_year]
    test_df = df[df['year'] == test_year]

    X_train = train_df.drop(columns=['playoff', 'confID'])
    y_train = train_df['playoff']

    X_test = test_df.drop(columns=['playoff', 'confID'])
    y_test = test_df['playoff']

    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    result = predict(model, X_test, y_test, test_df.copy())
    displayResults(result)

    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)
    plt.show()

def customGridSearch(df,model,param_grid, metric="accuracy", start_test_year=4,end_test_year=10):
    """
        Perform a customized Grid Search with a certain model and param_grid from year 4 to 10
    """
    running_percentage = 0
    best_score = -1.0
    best_grid = None
    for i,g in enumerate(ParameterGrid(param_grid)):
        running_percentage = i/len(ParameterGrid(param_grid))
        print(f"Grid Search is {round(running_percentage*100,2)}% done",end='\r')
        model.set_params(**g)
        best_avg_year_score = 0
        for year in range(start_test_year,end_test_year+1):
            res = runModel(df,model,year)
            match metric:
                case "accuracy":
                    best_avg_year_score += res.accuracy
                    break
                case "f1":
                    best_avg_year_score += res.f1
                    break
                case "precision":
                    best_avg_year_score += res.precision
                    break
                case "recall":
                    best_avg_year_score += res.recall
                    break
                case "avg":
                    avg = (res.accuracy + res.f1 + res.precision + res.recall)/4
                    best_avg_year_score += avg
                    break

        best_avg_year_score = best_avg_year_score / (end_test_year+1-start_test_year)

        # save if best
        if(best_score < best_avg_year_score):
            best_score = best_avg_year_score
            best_grid = g
    return best_grid, best_score

def DecisionTree_GridSearch(df, metric = "accuracy", start_test_year=4, end_test_year=10):
    """
        Perform a grid search for Decision Tree model
    """
    # grid search trees hyperparameters
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 10),
        'max_features': ['sqrt', 'log2']
    }
    model = DecisionTreeClassifier(random_state=42)
    best_grid,_ = customGridSearch(df,model,param_grid, metric, start_test_year, end_test_year)
    print(best_grid)
    return best_grid

def RandomForest_GridSearch(df, metric = "accuracy", start_test_year=4, end_test_year=10):
    """
        Perform a grid search for a Random Forest model
    """
    # grid search Random Forest's hyperparameters
    param_grid = {
        'n_estimators': [5*x for x in range(1,10)],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None] + list(range(1, 10)),
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_split': range(2, 10),
        # 'min_samples_leaf': range(1, 10),
    }
    model = RandomForestClassifier(random_state=42)
    best_grid, _ = customGridSearch(df,model,param_grid, metric, start_test_year, end_test_year)
    print(best_grid)
    return best_grid

def NeuralNet_GridSearch(df, metric = "accuracy", start_test_year=4, end_test_year=10):
    """
        Perform a grid search for a Neural Net model
    """
    # grid search Random Forest's hyperparameters
    param_grid = {
        'hidden_layer_sizes': [(5, 2), (100, 100, 100), (100, 100), (100, 100, 100, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.000001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [5000]
    }
    model = MLPClassifier(random_state=42)
    best_grid, _ = customGridSearch(df,model,param_grid, metric, start_test_year, end_test_year)
    print(best_grid)
    return best_grid


def SVM_GridSearch(df, metric = "accuracy", start_test_year=4, end_test_year=10):
    """
        Perform a grid search for a SVM model
    """
    # grid search Random Forest's hyperparameters
    param_grid = {
        'C': np.arange(0.2, 1.8, 0.2),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'probability': [False,True],
        'shrinking':[False,True]
    }
    model = SVC(random_state=42)
    best_grid, _ = customGridSearch(df,model,param_grid, metric, start_test_year, end_test_year)
    print(best_grid)
    return best_grid

def KNN_GridSearch(df, metric = "accuracy", start_test_year=4, end_test_year=10):
    """
        Perform a grid search for a KNN model
    """
    # grid search Random Forest's hyperparameters
    param_grid = {'n_neighbors': list(range(1, 30)),
                  'weights': ['uniform', 'distance'],
                   'p' : [1,2] }
    model = KNeighborsClassifier()
    best_grid, _ = customGridSearch(df, model, param_grid, metric, start_test_year, end_test_year)
    print(best_grid)
    return best_grid

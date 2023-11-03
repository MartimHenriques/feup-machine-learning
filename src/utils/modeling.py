from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import export_graphviz, plot_tree
import seaborn as sb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy import ndarray

@dataclass
class Result:
    y_test: ndarray
    y_pred: ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float

    def toRow(self): 
        return [self.accuracy, self.precision, self.recall, self.f1]

'''Get metrics for the prediction'''
def getMetrics(model, testing_inputs, testing_classes) -> Result:
    model.score(testing_inputs, testing_classes)

    y_pred = model.predict(testing_inputs)

    accuracy = accuracy_score(testing_classes, y_pred)
    precision = precision_score(testing_classes, y_pred)
    recall = recall_score(testing_classes, y_pred)
    f1 = f1_score(testing_classes, y_pred)
    return Result(testing_classes, y_pred, accuracy, precision, recall, f1)

'''Run a model and print results'''
def runModel(df, model, test_year=10):
    train_df = df[df['year'] < test_year]
    test_df = df[df['year'] == test_year]

    X_train = train_df.drop('playoff', axis=1)
    y_train = train_df['playoff']

    X_test = test_df.drop('playoff', axis=1)
    y_test = test_df['playoff']

    model.fit(X_train, y_train)
    return getMetrics(model, X_test, y_test)

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
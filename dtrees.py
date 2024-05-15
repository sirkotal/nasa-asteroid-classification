import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import data_preprocessing


def train_decision_trees():
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    dt_class = DecisionTreeClassifier(random_state=1)
    dt_class.fit(training_inputs, training_classes)

    dt_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, dt_class.predict(testing_inputs))

    print(confusion_matrix(testing_classes, dt_class.predict(testing_inputs)))

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(testing_classes, dt_class.predict(testing_inputs)))
    cm_display.plot()
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.xlabel('Predicted Hazard')
    plt.ylabel('Actual Hazard')
    plt.show()

    print(classification_report(testing_classes, dt_class.predict(testing_inputs)))

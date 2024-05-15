import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import data_preprocessing


def train_naive_bayes():
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    nb_class = GaussianNB()
    nb_class.fit(training_inputs, training_classes)

    nb_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, nb_class.predict(testing_inputs))

    print(confusion_matrix(testing_classes, nb_class.predict(testing_inputs)))

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(testing_classes, nb_class.predict(testing_inputs)))
    cm_display.plot()
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.xlabel('Predicted Hazard')
    plt.ylabel('Actual Hazard')
    plt.show()

    print(classification_report(testing_classes, nb_class.predict(testing_inputs)))

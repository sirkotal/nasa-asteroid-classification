import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import data_preprocessing


def train_ann(layer_n=25, act='relu', slv='adam'):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    ann_class = MLPClassifier(hidden_layer_sizes=(layer_n, layer_n, layer_n), activation=act, solver=slv, max_iter=200,
                              random_state=1)
    ann_class.fit(training_inputs, training_classes)

    ann_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, ann_class.predict(testing_inputs))

    print(confusion_matrix(testing_classes, ann_class.predict(testing_inputs)))

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(testing_classes, ann_class.predict(testing_inputs)))
    cm_display.plot()
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.xlabel('Predicted Hazard')
    plt.ylabel('Actual Hazard')
    plt.show()

    print(classification_report(testing_classes, ann_class.predict(testing_inputs)))

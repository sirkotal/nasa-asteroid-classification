import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import data_preprocessing, data_results


def train_ann(layer_n=25, act='relu', slv='adam'):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    ann_class = MLPClassifier(hidden_layer_sizes=(layer_n*4, layer_n*2, layer_n), activation=act, solver=slv,
                              max_iter=200, random_state=1)
    ann_class.fit(training_inputs, training_classes)

    ann_class.score(testing_inputs, testing_classes)

    # train_score = ann_class.score(training_inputs, training_classes)
    # test_score = ann_class.score(testing_inputs, testing_classes)
    # print(f'Training accuracy: {train_score}')
    # print(f'Testing accuracy: {test_score}')

    accuracy_score(testing_classes, ann_class.predict(testing_inputs))

    data_results(testing_classes, testing_inputs, ann_class)

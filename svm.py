from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import data_preprocessing, data_results


def train_svm(k='rbf'):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    svm_class = SVC(kernel=k)
    svm_class.fit(training_inputs, training_classes)

    svm_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, svm_class.predict(testing_inputs))

    data_results(testing_classes, testing_inputs, svm_class)

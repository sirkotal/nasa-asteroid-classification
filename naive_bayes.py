from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from utils import data_preprocessing, data_results


def train_naive_bayes():
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    nb_class = GaussianNB()
    nb_class.fit(training_inputs, training_classes)

    nb_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, nb_class.predict(testing_inputs))

    data_results(testing_classes, testing_inputs, nb_class)

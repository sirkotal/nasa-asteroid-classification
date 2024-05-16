from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import data_preprocessing, data_results


def train_rand_forest(n=100):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    rf_class = RandomForestClassifier(n_estimators=n)
    rf_class.fit(training_inputs, training_classes)

    rf_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, rf_class.predict(testing_inputs))

    data_results(testing_classes, testing_inputs, rf_class)

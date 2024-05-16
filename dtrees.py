from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import data_preprocessing, data_results


def train_decision_trees():
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    dt_class = DecisionTreeClassifier(random_state=1)
    dt_class.fit(training_inputs, training_classes)

    dt_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, dt_class.predict(testing_inputs))

    data_results(testing_classes, testing_inputs, dt_class)

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from utils import data_preprocessing, data_results


def train_knn(n=5):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = data_preprocessing()

    knn_class = neighbors.KNeighborsClassifier(n_neighbors=n)
    knn_class.fit(training_inputs, training_classes)

    knn_class.score(testing_inputs, testing_classes)

    # train_score = knn_class.score(training_inputs, training_classes)
    # test_score = knn_class.score(testing_inputs, testing_classes)
    # print(f'Training accuracy: {train_score}')
    # print(f'Testing accuracy: {test_score}')

    accuracy_score(testing_classes, knn_class.predict(testing_inputs))

    data_results(testing_classes, testing_inputs, knn_class)

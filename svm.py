import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def train_svm(k='linear'):
    # Load the data
    data = pd.read_csv('nasa.csv')

    if data.isna().any().any():
        data.fillna("NA", inplace=True)

    print(data.head())

    nasa_class = data.pop('Hazardous')
    data.insert(0, 'Hazardous', nasa_class)

    data.drop(columns=['Close Approach Date', 'Orbit Determination Date'], inplace=True)

    if (data['Equinox'].nunique() == 1):
        data.drop(columns=['Equinox'], inplace=True)
    if (data['Orbiting Body'].nunique() == 1):
        data.drop(columns=['Orbiting Body'], inplace=True)

    print(data.dtypes)
    stats = data.describe()
    print(stats)

    # print(sb.pairplot(data.dropna(), hue='Hazardous'))

    label_encoder = LabelEncoder()

    new_data = data.drop('Hazardous', axis=1)
    hazard = data['Hazardous']
    hazard = label_encoder.fit_transform(hazard)

    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(new_data, hazard, test_size=0.25, random_state=1)

    svm_class = SVC(kernel=k)
    svm_class.fit(training_inputs, training_classes)

    svm_class.score(testing_inputs, testing_classes)

    accuracy_score(testing_classes, svm_class.predict(testing_inputs))

    print(confusion_matrix(testing_classes, svm_class.predict(testing_inputs)))

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(testing_classes, svm_class.predict(testing_inputs)))
    cm_display.plot()
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.xlabel('Predicted Hazard')
    plt.ylabel('Actual Hazard')
    plt.show()

    print(classification_report(testing_classes, svm_class.predict(testing_inputs)))

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def data_preprocessing():
    # Load the data
    data = pd.read_csv('data/nasa.csv')

    if data.isna().any().any():
        data.fillna("NA", inplace=True)

    print(data.head())

    nasa_class = data.pop('Hazardous')
    data.insert(0, 'Hazardous', nasa_class)

    data.drop(columns=['Close Approach Date', 'Orbit Determination Date'], inplace=True)

    # keep SI units (or the ones closer to them)
    data.drop(columns=['Est Dia in KM(min)', 'Est Dia in KM(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
                       'Est Dia in Feet(min)', 'Est Dia in Feet(max)'], inplace=True)
    data.drop(columns=['Relative Velocity km per hr', 'Miles per hour'], inplace=True)
    data.drop(columns=['Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)'], inplace=True)

    if data['Equinox'].nunique() == 1:
        data.drop(columns=['Equinox'], inplace=True)
    if data['Orbiting Body'].nunique() == 1:
        data.drop(columns=['Orbiting Body'], inplace=True)

    print(data.dtypes)
    stats = data.describe()
    print(stats)

    # print(sb.pairplot(data.dropna(), hue='Hazardous'))

    label_encoder = LabelEncoder()

    new_data = data.drop('Hazardous', axis=1)
    hazard = data['Hazardous']
    hazard = label_encoder.fit_transform(hazard)

    return train_test_split(new_data, hazard, test_size=0.25, random_state=1)


def data_results(testing_classes, testing_inputs, alg_class):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(testing_classes, alg_class.predict(testing_inputs))
    )

    cm_display.plot()
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.xlabel('Predicted Hazard')
    plt.ylabel('Actual Hazard')
    plt.show()

    print(classification_report(testing_classes, alg_class.predict(testing_inputs)))


def display_menu():
    print("---------------------------- NASA Asteroid Classification ----------------------------")
    print("")
    print("Please choose a classifier:")
    print("")
    print("1. Decision Trees")
    print("2. K-Nearest Neighbors")
    print("3. Support Vector Machines")
    print("4. Artificial Neural Network")
    print("5. Naive Bayes")
    print("6. Random Forest")
    print("")
    print("i. Disclaimer")
    print("")
    print("0. Exit")
    print("")

    choice = input("> ")
    if choice not in ['0', '1', '2', '3', '4', '5', '6', 'i']:
        raise ValueError("Invalid choice")

    return choice

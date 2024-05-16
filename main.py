import sys
from classifiers.dtrees import train_decision_trees
from classifiers.knn import train_knn
from classifiers.rand_forest import train_rand_forest
from classifiers.svm import train_svm
from classifiers.neural import train_ann
from classifiers.naive_bayes import train_naive_bayes
from utils import display_menu


def main():
    while True:
        user_choice = display_menu()
        print("")
        if user_choice == '1':  # Decision Trees
            train_decision_trees()
        elif user_choice == '2':  # KNN
            train_knn()
        elif user_choice == '3':  # SVM
            train_svm()
        elif user_choice == '4':  # ANN
            train_ann()
        elif user_choice == '5':  # NB
            train_naive_bayes()
        elif user_choice == '6':  # RF
            train_rand_forest()
        elif user_choice == 'i':  # Info
            print("<-------------------------------------------------------->")
            print("The only classifiers that seem to suit this dataset are Decision Trees and Random Forest.")
            print("We made other classifiers available, but they may not yield proper results.")
            print("<-------------------------------------------------------->")
        elif user_choice == '0':
            sys.exit(1)
        else:
            print("")
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

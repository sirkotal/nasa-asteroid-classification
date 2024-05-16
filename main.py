import sys
from classifiers.dtrees import train_decision_trees
from classifiers.knn import train_knn
from classifiers.rand_forest import train_rand_forest
from classifiers.svm import train_svm
from classifiers.neural import train_ann
from classifiers.naive_bayes import train_naive_bayes


def main():
    #if len(sys.argv) != 3:
    #    print("Invalid Arguments")
    #    sys.exit(1)

    # algorithm = sys.argv[1]
    algorithm = "ann"

    if algorithm == 'decision_trees':
        train_decision_trees()
    elif algorithm == 'knn':
        train_knn()
    elif algorithm == 'svm':
        train_svm()
    elif algorithm == 'ann':
        train_ann()
    elif algorithm == 'nb':
        train_naive_bayes()
    elif algorithm == 'rf':
        train_rand_forest()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

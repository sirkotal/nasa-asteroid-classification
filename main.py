import sys
from dtrees import train_decision_trees
from knn import train_knn


def main():
    #if len(sys.argv) != 3:
    #    print("Invalid Arguments")
    #    sys.exit(1)

    # algorithm = sys.argv[1]
    algorithm = "knn"

    if algorithm == 'decision_trees':
        train_decision_trees()
    elif algorithm == 'knn':
        train_knn()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
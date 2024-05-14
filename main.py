import sys
from dtrees import train_decision_trees


def main():
    #if len(sys.argv) != 3:
    #    print("Invalid Arguments")
    #    sys.exit(1)

    # algorithm = sys.argv[1]
    algorithm = "decision_trees"

    if algorithm == 'decision_trees':
        train_decision_trees()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
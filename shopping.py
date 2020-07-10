import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)
    
    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
        }
    evidence = []
    labels = []

    with open(filename) as data_file:
        data = csv.reader(data_file)
        next(data)

        for row in data:
            evidence_row = []
            n = 0 
            for cell in row[:17]:
                if n in [0, 2, 4, 11, 12, 13, 14]:
                    evidence_row.append(int(cell))
                if n in [1, 3, 5, 6, 7, 8, 9]:
                    evidence_row.append(float(cell))
                if n == 10:
                    evidence_row.append(months[cell])
                if n == 15:
                    if cell == "Returning_Visitor":
                        evidence_row.append(1)
                    else: 
                        evidence_row.append(0)
                if n == 16:
                    if cell == "True":
                        evidence_row.append(1)
                    else:
                        evidence_row.append(0)
                n += 1
            evidence.append(evidence_row)
            if row[17] == "TRUE":
                labels.append(1)
            else:
                labels.append(0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    actual_positive = labels.count(1)
    actual_negative = labels.count(0)

    positive_identified = 0
    negative_identified = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            positive_identified += 1
        if actual == 0 and predicted == 0:
            negative_identified += 1

    sensitivity = positive_identified / actual_positive
    specificity = negative_identified / actual_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

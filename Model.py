from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit


def buildModel(X, y):
    # Define the model
    model = LogisticRegression()

    # Define the splitter for splitting the data in a train set and a test set
    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=0)

    # Loop through the splits (only one)
    for train_index, test_index in splitter.split(X, y):
        # Select the train and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Fit and predict!
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # show the results
    print(classification_report(y_test, y_pred))

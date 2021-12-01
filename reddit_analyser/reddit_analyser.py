from scraper import Scraper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
import numpy as np


def get_data(use_local: bool = True) -> (dict, dict, dict):
    """
    Obtain the data for processing by either loading it from the local
    directory or by creating a reddit scraper to fetch everything we need
    """
    s = Scraper()
    if not use_local:
        s.pull_data()
        s.dump_data()
        return s.data
    else:
        return s.load_data()


#########################################################################
"""
Place functions pertaining to the various ML approaches used here,
as well as the functions defined to perform analysis on their performance
"""

def temp_logistic_regression(x_data, y_data, dictionary) -> None:
    # x = data.x #TODO
    # y = data.y #TODO
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # model = LogisticRegression()
    # model.fit(x_train, y_train)
    # predicted_x = model.predict(x_test)
    # fpr, tpr, thresholds = roc_curve(y_test, predicted_x)
    # score = auc(fpr, tpr)

    for sub in x_data.keys():
        x = np.array(x_data[sub])
        y = np.array([[val] for val in y_data[sub]])
        print(len(x), len(y))
        counts = [sum(x[:, i]) for i in range(len(x[0]))]
        highest = max(counts)
        print(
            f"Most popular word: {dictionary[sub][counts.index(highest)]} with {highest}"
        )

#########################################################################


def main():
    """
    Should only be used to pass around the data to the various functions
    which will process it
    """
    x, y, d = get_data(use_local=False)
    temp_logistic_regression(x, y, d)


if __name__ == "__main__":
    main()

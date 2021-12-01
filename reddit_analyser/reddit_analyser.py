from scraper import Scraper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
import numpy as np


def get_data(use_local: bool = True) -> (list, dict):
    """
    Obtain the data for processing by either loading it from the local
    directory or by creating a reddit scraper to fetch everything we need
    """
    if not use_local:
        scraper = Scraper()
        scraper.pull_data()
        scraper.dump_data()
        return scraper.data, scraper.dictionary
    else:
        return


#########################################################################
"""
Place functions pertaining to the various ML approaches used here,
as well as the functions defined to perform analysis on their performance
"""


def temp_logistic_regression(data, dictionary) -> None:
    # x = data.x #TODO
    # y = data.y #TODO
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # model = LogisticRegression()
    # model.fit(x_train, y_train)
    # predicted_x = model.predict(x_test)
    # fpr, tpr, thresholds = roc_curve(y_test, predicted_x)
    # score = auc(fpr, tpr)

    counts = [sum(data[:, i]) for i in range(len(data[0]))]
    highest = max(counts)
    print(
        f"Most popular word: {dictionary['ProgrammerHumor'][counts.index(highest)]} with {highest}"
    )


#########################################################################


def main():
    """
    Should only be used to pass around the data to the various functions
    which will process it
    """
    data, dictionary = get_data(use_local=False)
    data = np.array(data)
    temp_logistic_regression(data, dictionary)


if __name__ == "__main__":
    main()

from scraper import Scraper
import numpy as np


def get_data(use_local=True):
    """
    Obtain the data for processing by either loading it from the local
    directory or by creating a reddit scraper to fetch everything we need
    """
    if not use_local:
        s = Scraper()
        s.pull_data()
        s.dump_data()
        return s.x_data, s.y_data, s.dictionary
    else:
        return


#########################################################################
"""
Place functions pertaining to the various ML approaches used here,
as well as the functions defined to perform analysis on their performance
"""


def example_processing(x_data, y_data, dictionary):
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
    example_processing(x, y, d)


if __name__ == "__main__":
    main()

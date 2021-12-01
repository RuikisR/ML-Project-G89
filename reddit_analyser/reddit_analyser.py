from scraper import Scraper
import numpy as np


def get_data(use_local=True):
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


def example_processing(data, dictionary):
    counts = [sum(data[:, i]) for i in range(len(data[0]))]
    highest = max(counts)
    print(f"Most popular word: {dictionary[counts.index(highest)]} with {highest}")


#########################################################################


def main():
    """
    Should only be used to pass around the data to the various functions
    which will process it
    """
    data, dictionary = get_data(use_local=False)
    data = np.array(data)
    example_processing(data, dictionary)


if __name__ == "__main__":
    main()

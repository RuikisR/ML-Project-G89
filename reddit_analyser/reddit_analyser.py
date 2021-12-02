"""
Module to run the reddit scraper and data analysis modules
"""
from scraper import Scraper as S
from data_analysis import DataAnalyser as DA


def get_data(use_local: bool = True) -> (dict, dict, dict):
    """
    Obtain the data for processing by either loading it from the local
    directory or by creating a reddit scraper to fetch everything we need
    """
    scraper = S()
    if not use_local:
        scraper = S()
        scraper.pull_data()
        scraper.dump_data()
        return scraper.data
    return scraper.load_data()


def main():
    """
    Should only be used to pass around the data to the various functions
    which will process it
    """
    x_data, y_data, dictionary = get_data(use_local=False)
    data_analysis = DA()
    data_analysis.dummy_classifier(x_data, y_data, dictionary)


if __name__ == "__main__":
    main()

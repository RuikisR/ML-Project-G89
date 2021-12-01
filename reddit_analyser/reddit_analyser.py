from scraper import Scraper as S
from data_analysis import DataAnalyser as DA


def get_data(use_local: bool = True) -> (dict, dict, dict):
    """
    Obtain the data for processing by either loading it from the local
    directory or by creating a reddit scraper to fetch everything we need
    """
    s = S()
    if not use_local:
        s = S()
        s.pull_data()
        s.dump_data()
        return s.data
    else:
        return s.load_data()


def main():
    """
    Should only be used to pass around the data to the various functions
    which will process it
    """
    x, y, d = get_data(use_local=False)
    da = DA()
    da.temp_logistic_regression(x, y, d)


if __name__ == "__main__":
    main()

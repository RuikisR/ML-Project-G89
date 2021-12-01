from scraper import Scraper


def get_data(use_local=True):
    """
    Obtain the data for processing by either loading it from the local
    directory or by creating a reddit scraper to fetch everything we need
    """
    if not use_local:
        scraper = Scraper()
        data = scraper.pull_data()
        scraper.dump_data()
        return data
    else:
        return


#########################################################################
"""
Place functions pertaining to the various ML approaches used here,
as well as the functions defined to perform analysis on their performance
"""


def example_processing(data):
    [print(line) for line in data]


#########################################################################


def main():
    """
    Should only be used to pass around the data to the various functions
    which will process it
    """
    data = get_data(use_local=False)
    # example_processing(data)


if __name__ == "__main__":
    main()

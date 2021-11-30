import scraper


def main():
    reddit = scraper.create_instance()
    [print(line) for line in scraper.pull_data(reddit)]


if __name__ == '__main__':
    main()

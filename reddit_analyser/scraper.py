import praw


class Scraper:
    def __init__(self):
        """
        Init function to instantiate our praw.reddit class and other class specific data
        """
        self.instance = praw.Reddit("scraper", user_agent="scraper user agent")
        self.data = []

    def pull_data(self):
        """
        Pulls our raw data from the Reddit API
        (Will likely use a config file to control this)
        """
        self.data = [
            post.title for post in self.instance.subreddit("dankmemes").hot(limit=10)
        ]

    def dump_data(self):
        """
        Function to dump our data into a csv file contained in the data directory
        """
        pass

    def parse_data(self):
        """
        Function to segment our data into a usable state
        """
        pass

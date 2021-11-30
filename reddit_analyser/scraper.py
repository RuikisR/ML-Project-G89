import praw


class Scraper:
    def __init__(self):
        self.instance = praw.Reddit("scraper", user_agent="scraper user agent")

    def pull_data(self):
        """
        Pulls the data we want from the Reddit API
        (Will likely use a config file to control this)
        """
        data_gen = self.instance.subreddit("dankmemes").hot(limit=10)
        self.data = [post.title for post in data_gen]

    def dump_data(self):
        """
        Used to dump the data into a csv file contained in the data/ directory
        """
        pass

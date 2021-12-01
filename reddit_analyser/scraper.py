import praw
import yaml


class Scraper:

    MEANINGLESS_CHARS = "1234567890`¬-=_+[]{};'#:@~,./<>?\\|!\"£$%^&*()"

    def __init__(self):
        """
        Init function to instantiate our praw.reddit class and other class specific data
        """
        self.instance = praw.Reddit("scraper", user_agent="scraper user agent")
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
        self.data = []
        self.dictionary = {}

    def pull_data(self):
        """
        Pulls our raw data from the Reddit API
        (Will likely use a config file to control this)
        """
        for s in self.config["subreddits"]:
            posts = self.instance.subreddit(s).hot(limit=1)
            for post in posts:
                post.comments.replace_more(limit=None)
                self.parse_data(post.title, s)
                for comment in post.comments.list():
                    self.parse_data(comment.body, s)
        self.pad_data()

    def parse_data(self, input_string: str, current_subreddit: str):
        """
        Function to segment our data into a usable state
        """
        self.dictionary[current_subreddit] = []
        input_string = input_string.lower()
        for c in Scraper.MEANINGLESS_CHARS:
            input_string = input_string.replace(c, "")
        words = input_string.split(" ")
        for word in words:
            if word != "" and word not in self.dictionary:
                self.dictionary[current_subreddit].append(word)
        self.data.append(
            [words.count(word) for word in self.dictionary[current_subreddit]]
        )

    def pad_data(self):
        max_len = max([len(entry) for entry in self.data])
        for entry in self.data:
            for _ in range(max_len - len(entry)):
                entry.append(0)

    def dump_data(self):
        """
        Function to dump our data into a csv file contained in the data directory
        """
        pass

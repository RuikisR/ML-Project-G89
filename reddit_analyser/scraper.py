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
        self.x_data = {}
        self.y_data = {}
        self.dictionary = {}

    def pull_data(self):
        """
        Pulls our raw data from the Reddit API
        (Will likely use a config file to control this)
        """
        for s in self.config["subreddits"]:
            self.dictionary[s] = []
            self.x_data[s] = []
            self.y_data[s] = []
            posts = self.instance.subreddit(s).hot(limit=10)
            for post in posts:
                post.comments.replace_more(limit=None)
                self.parse_data(post.title, s)
                self.y_data[s].append(post.score)
                for comment in post.comments.list():
                    self.parse_data(comment.body, s)
                    self.y_data[s].append(comment.score)
        self.pad_data()

    def parse_data(self, input_string: str, current_subreddit: str):
        """
        Function to segment our data into a usable state
        """
        input_string = input_string.lower()
        for c in Scraper.MEANINGLESS_CHARS:
            input_string = input_string.replace(c, "")
        words = input_string.split(" ")
        for word in words:
            if word != "" and word not in self.dictionary[current_subreddit]:
                self.dictionary[current_subreddit].append(word)
        self.x_data[current_subreddit].append(
            [words.count(word) for word in self.dictionary[current_subreddit]]
        )

    def pad_data(self):
        for x in self.x_data.keys():
            max_len = max([len(entry) for entry in self.x_data[x]])
            for entry in self.x_data[x]:
                for _ in range(max_len - len(entry)):
                    entry.append(0)

    def dump_data(self):
        """
        Function to dump our data into a csv file contained in the data directory
        """
        pass

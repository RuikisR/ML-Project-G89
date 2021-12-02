"""
Module to contain Scraper class
"""
import pickle
from os.path import join
import logging
import praw
import yaml


class Scraper:
    """
    Class to handle data collection and data preparation
    """

    def __init__(self):
        """
        Init function to instantiate our praw.reddit class and other class specific data
        """
        self.instance = praw.Reddit("scraper", user_agent="scraper user agent")
        with open("config.yaml", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.submission_limit = self.config["submission_limit"]
        self.subreddits = self.config["subreddits"]
        self.x_data = {}
        self.y_data = {}
        self.dictionary = {}
        self.data_path = join(self.config["data_dir"], self.config["data_file_name"])
        if self.config["debug"]:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(
                level=logging.WARNING,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

    @property
    def data(self):
        """
        Property to return current scraper data
        """
        return self.x_data, self.y_data, self.dictionary

    def pull_data(self):
        """
        Pulls our raw data from the Reddit API
        """
        for i, subreddit in enumerate(self.config["subreddits"]):
            logging.info(
                "Pulling from subreddit 'r/%s' (%d of %d)",
                subreddit,
                i + 1,
                len(self.subreddits),
            )
            self.dictionary[subreddit] = []
            self.x_data[subreddit] = []
            self.y_data[subreddit] = []
            posts = self.instance.subreddit(subreddit).top(
                "all", limit=self.submission_limit
            )

            for j, post in enumerate(posts):
                logging.info(
                    "Beginning processing for submission %d of %d",
                    j + 1,
                    self.submission_limit,
                )
                self.parse_data(post.title, subreddit)
                self.y_data[subreddit].append(post.score)
                logging.info(
                    "Expanding comments for submission %d of %d",
                    j + 1,
                    self.submission_limit,
                )
                post.comments.replace_more(limit=None)
                logging.info(
                    "Parsing comments for submission %d of %d",
                    j + 1,
                    self.submission_limit,
                )
                for comment in post.comments.list():
                    self.parse_data(comment.body, subreddit)
                    self.y_data[subreddit].append(comment.score)
        self.pad_data()

    def parse_data(self, input_string: str, current_subreddit: str):
        """
        Function to segment our data into a usable state
        """
        input_string = input_string.lower()
        for chars in self.config["meaningless_chars"]:
            input_string = input_string.replace(chars, "")
        words = input_string.split(" ")
        for word in words:
            if word != "" and word not in self.dictionary[current_subreddit]:
                self.dictionary[current_subreddit].append(word)
        self.x_data[current_subreddit].append(
            [words.count(word) for word in self.dictionary[current_subreddit]]
        )

    def pad_data(self):
        """
        Function that ensures that all data lists are off equal length
        """
        for curent_x in self.x_data:
            max_len = max([len(entry) for entry in self.x_data[curent_x]])
            for entry in self.x_data[curent_x]:
                for _ in range(max_len - len(entry)):
                    entry.append(0)

    def dump_data(self):
        """
        Function to pickle our data for later use
        """
        with open(self.data_path, "wb") as file:
            pickle.dump((self.x_data, self.y_data, self.dictionary), file)

    def load_data(self):
        """
        Unpickles data which has already been saved
        """
        with open(self.data_path, "rb") as file:
            return pickle.load(file)

import praw


class Scraper:

    MEANINGLESS_CHARS = "1234567890`¬-=_+[]{};'#:@~,./<>?\\|!\"£$%^&*()"

    def __init__(self):
        """
        Init function to instantiate our praw.reddit class and other class specific data
        """
        self.instance = praw.Reddit("scraper", user_agent="scraper user agent")
        self.data = []
        self.dictionary = []

    def pull_data(self):
        """
        Pulls our raw data from the Reddit API
        (Will likely use a config file to control this)
        """
        posts = self.instance.subreddit("ProgrammerHumor").hot(limit=10)
        for post in posts:
            post.comments.replace_more(limit=None)
            self.parse_data(post.title)
            for comment in post.comments.list():
                self.parse_data(comment.body)
        self.pad_data()
        [print(len(entry)) for entry in self.data]

    def parse_data(self, words):
        """
        Function to segment our data into a usable state
        """
        words = words.lower()
        for c in Scraper.MEANINGLESS_CHARS:
            words = words.replace(c, "")
        words = words.split(" ")
        for word in words:
            if word not in self.dictionary:
                self.dictionary.append(word)
        self.data.append([words.count(word) for word in self.dictionary])

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

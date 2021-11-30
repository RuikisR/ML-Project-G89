import praw


def create_instance():
    return praw.Reddit("scraper", user_agent="scraper user agent")


def pull_data(instance):
    return [post.title for post in instance.subreddit("dankmemes").hot(limit=10)]

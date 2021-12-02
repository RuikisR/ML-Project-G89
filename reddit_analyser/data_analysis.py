"""
Module to contain Machine Learning functions
"""
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.dummy import DummyClassifier
import yaml
import numpy as np
import logging


class DataAnalyser:
    """
    Class to handle all required Machine learning on reddit data
    """

    def __init__(self):
        with open("config.yaml", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.const = self.config["analyser_config"]
        if self.config["debug"]:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(
                level=logging.WARNING,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

    def lasso_regression(self, x_data: dict, y_data: dict):
        """
        Function to perform Lasso Regression on our data per subreddit
        """
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model = Lasso(alpha=self.const["lasso_alpha"])
            model.fit(x_train, y_train)
            predicted_x = model.predict(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, predicted_x)
            score = auc(fpr, tpr)
            logging.info("Lasso Regression on subreddit %s scored: %s", subreddit, score)

    def logistic_regression(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform Logistic Regression on our data per subreddit
        """
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model = LogisticRegression()
            model.fit(x_train, y_train)
            predicted_x = model.predict(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, predicted_x)
            score = auc(fpr, tpr)
            logging.info("Logistic Regression on subreddit %s scored: %s", subreddit, score)

    def k_neighbors(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform K Neighbors on our data per subreddit
        """
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model = KNeighborsClassifier(n_neighbors=self.const["K_val"])
            model.fit(x_train, y_train)
            score = cross_val_score(model, x_test, y_test, cv=self.const["cross_val"])
            logging.info("K Nearest Neighbours on subreddit %s scored: %s", subreddit, score)

    def dummy_classifier(self, x_data: dict, y_data: dict, dictionary):
        """
        Function to perform naive classification on our data per subreddit
        """
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model = DummyClassifier(strategy=self.const["dummy_strategy"])
            model.fit(x_train, y_train)
            score = cross_val_score(model, x_test, y_test, cv=self.const["cross_val"])
            logging.info("Dummy Classifier on subreddit %s scored: %s", subreddit, score)

        for sub in x_data.keys():
            x_val = np.array(x_data[sub])
            y_val = np.array([[val] for val in y_data[sub]])
            logging.info("%s %s", len(x_val), len(y_val))
            counts = [sum(x_val[:, i]) for i in range(len(x_val[0]))]
            highest = max(counts)
            logging.info(
                "Most popular word: '%s' with %s occurrences",
                dictionary[sub][counts.index(highest)],
                highest,
            )


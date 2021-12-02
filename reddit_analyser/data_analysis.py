"""
Module to contain Machine Learning functions
"""
import logging
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.dummy import DummyClassifier
import yaml
import numpy as np


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

    def model_execution(self, x_data: dict, y_data: dict, model_name: str, model):
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model.fit(x_train, y_train)
            fpr, tpr, thresholds = roc_curve(y_test, model.predict(x_test))
            score = auc(fpr, tpr)
            logging.info("%s on subreddit %s scored: %s", model_name, subreddit, score)

    def lasso_regression(self, x_data: dict, y_data: dict):
        """
        Function to perform Lasso Regression on our data per subreddit
        """
        self.model_execution(
            x_data, y_data, "Lasso Regression", Lasso(alpha=self.const["lasso_alpha"])
        )

    def logistic_regression(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform Logistic Regression on our data per subreddit
        """
        self.model_execution(
            x_data, y_data, "Logistic Regression", LogisticRegression()
        )

    def k_neighbors(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform K Neighbors on our data per subreddit
        """
        self.model_execution(
            x_data,
            y_data,
            "K Nearest Neighbours",
            KNeighborsClassifier(n_neighbors=self.const["K_val"]),
        )

    def dummy_classifier(self, x_data: dict, y_data: dict, dictionary):
        """
        Function to perform naive classification on our data per subreddit
        """
        self.model_execution(
            x_data,
            y_data,
            "Dummy Classifier",
            DummyClassifier(strategy=self.const["dummy_strategy"]),
        )

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

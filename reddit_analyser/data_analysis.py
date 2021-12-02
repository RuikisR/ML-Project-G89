"""
Module to contain Machine Learning functions
"""
import logging
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, confusion_matrix
from sklearn.dummy import DummyClassifier
import yaml
import numpy as np
import matplotlib.pyplot as plt


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
        """
        Function to execute the input model and log it's output
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        :param model_name: The name of the model to be executed
        :param model: The model class to be used in our predictions
        """
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            for y_val in range(len(y_list)):
                if y_list[y_val] < 10:
                    y_list[y_val] = 0
                else:
                    y_list[y_val] = 1
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model.fit(x_train, y_train)
            predicted_values = model.predict(x_test)
            logging.info("%s confusion matrix: ", repr(str(confusion_matrix(y_test, predicted_values))))

            fpr, tpr, thresholds = roc_curve(y_test, predicted_values)
            score = auc(fpr, tpr)
            logging.info("%s on subreddit %s got an auc score: %s", model_name, subreddit, score)

            # TODO: decide what we want to do with these
            # polynomial_features = PolynomialFeatures(degree=2)
            # polynomial_train = polynomial_features.fit_transform(x_train)
            # polynomial_test = polynomial_features.fit_transform(x_test)

            # score = cross_val_score(model, x_test, y_test, cv=self.const["cross_val"])
            # logging.info("%s on subreddit %s got a cross validation score: %s", model_name, subreddit, score)

            RocCurveDisplay.from_estimator(model, x_test, y_test)
            plt.show()

    def lasso_regression(self, x_data: dict, y_data: dict):
        """
        Function to perform Lasso Regression on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        self.model_execution(
            x_data, y_data, "Lasso Regression", Lasso(alpha=self.const["lasso_alpha"])
        )

    def logistic_regression(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform Logistic Regression on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        self.model_execution(
            x_data, y_data, "Logistic Regression", LogisticRegression()
        )

    def k_neighbors(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform K Neighbors on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
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
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        :param dictionary: temp dictionary of word counts
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

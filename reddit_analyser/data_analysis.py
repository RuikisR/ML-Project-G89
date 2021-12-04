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
        metrics = {}
        for subreddit in self.config["subreddits"]:
            x_list = np.array(x_data[subreddit])
            y_list = np.array(y_data[subreddit])
            for y_val in range(len(y_list)):
                if y_list[y_val] < self.const["positive_reaction"]:
                    y_list[y_val] = 0
                else:
                    y_list[y_val] = 1
            x_train, x_test, y_train, y_test = train_test_split(
                x_list, y_list, test_size=self.const["test_size"]
            )
            model.fit(x_train, y_train)
            predicted_values = model.predict(x_test)

            fpr, tpr, thresholds = roc_curve(y_test, predicted_values)
            score = auc(fpr, tpr)
            logging.info(
                "%s on subreddit %s got an auc score: %s", model_name, subreddit, score
            )

            metrics[subreddit] = {
                "x_test": x_test,
                "y_test": y_test,
                "predicted_values": predicted_values,
                "trained_model": model,
                "score": score,
            }
        return metrics

    def lasso_regression(self, x_data: dict, y_data: dict):
        """
        Function to perform Lasso Regression on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        model_metrics = self.model_execution(
            x_data, y_data, "Lasso Regression", Lasso(alpha=self.const["lasso_alpha"])
        )
        for subreddit in self.config["subreddits"]:
            logging.info(
                "%s coefficient %s intercept %s",
                subreddit,
                model_metrics[subreddit]["trained_model"].coef_,
                model_metrics[subreddit]["trained_model"].intercept_,
            )

    def logistic_regression(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform Logistic Regression on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        for c_val in self.const["C_vals"]:
            model_metrics = self.model_execution(
                x_data,
                y_data,
                f"Logistic Regression with L:{c_val}",
                LogisticRegression(C=c_val),
            )

            for subreddit in self.config["subreddits"]:
                logging.info(
                    "%s confusion matrix: ",
                    repr(
                        str(
                            confusion_matrix(
                                model_metrics[subreddit]["y_test"],
                                model_metrics[subreddit]["predicted_values"],
                            )
                        )
                    ),
                )
                score = cross_val_score(
                    model_metrics[subreddit]["trained_model"],
                    model_metrics[subreddit]["x_test"],
                    model_metrics[subreddit]["y_test"],
                    cv=self.const["cross_val"],
                )
                logging.info(
                    "Logistic Regression with L:%s on subreddit %s got a cross validation score: %s",
                    c_val,
                    subreddit,
                    score,
                )

    def k_neighbors(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform K Neighbors on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        for k_val in self.const["K_vals"]:
            model_metrics = self.model_execution(
                x_data,
                y_data,
                f"K Nearest Neighbours with K: {k_val}",
                KNeighborsClassifier(n_neighbors=k_val),
            )

    def dummy_classifier(self, x_data: dict, y_data: dict, dictionary):
        """
        Function to perform naive classification on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        :param dictionary: temp dictionary of word counts
        """
        model_metrics = self.model_execution(
            x_data,
            y_data,
            "Dummy Classifier",
            DummyClassifier(strategy=self.const["dummy_strategy"]),
        )
        for subreddit in self.config["subreddits"]:
            logging.info(
                "%s confusion matrix: %s",
                subreddit,
                repr(
                    str(
                        confusion_matrix(
                            model_metrics["y_test"], model_metrics["predicted_values"]
                        )
                    )
                ),
            )

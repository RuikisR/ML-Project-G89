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
            # TODO ARBITRARY
            arbitrary_choice = 5
            for y_val in range(len(y_list)):
                if y_list[y_val] < arbitrary_choice:
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

            # TODO: decide what we want to do with these
            # polynomial_features = PolynomialFeatures(degree=2)
            # polynomial_train = polynomial_features.fit_transform(x_train)
            # polynomial_test = polynomial_features.fit_transform(x_test)

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
                "Lasso Regression on subreddit %s got an auc score: %s",
                subreddit,
                model_metrics[subreddit]["score"],
            )

    def logistic_regression(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform Logistic Regression on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        # c_values = [0.001, 0.01, 0.1, 0.5, 1]
        # for c_val in c_values:
        model_metrics = self.model_execution(
            x_data,
            y_data,
            f"Logistic Regression with L:{1}",
            LogisticRegression(C=1),
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
                1,
                subreddit,
                score,
            )
            RocCurveDisplay.from_estimator(
                model_metrics[subreddit]["trained_model"],
                model_metrics[subreddit]["x_test"],
                model_metrics[subreddit]["y_test"],
            )
            plt.show()

    def k_neighbors(self, x_data: dict, y_data: dict) -> None:
        """
        Function to perform K Neighbors on our data per subreddit
        :param x_data: dictionary containing the x_values indexed by subreddit name
        :param y_data: dictionary containing the y_values indexed by subreddit name
        """
        #k_values = [self.const["K_val"], 11, 17, 23]
        #for k_val in k_values:
        model_metrics = self.model_execution(
            x_data,
            y_data,
            f"K Nearest Neighbours with K: {self.const['K_val']}",
            KNeighborsClassifier(n_neighbors=self.const["K_val"]),
        )
        for subreddit in self.config["subreddits"]:
            logging.info(
                "%s confusion matrix: %s",
                subreddit,
                repr(
                    str(
                        confusion_matrix(
                            model_metrics[subreddit]["y_test"], model_metrics[subreddit]["predicted_values"]
                        )
                    )
                )
            )
            RocCurveDisplay.from_estimator(
                model_metrics[subreddit]["trained_model"],
                model_metrics[subreddit]["x_test"],
                model_metrics[subreddit]["y_test"],
            )
            plt.show()

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
                )
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

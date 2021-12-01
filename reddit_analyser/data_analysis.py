"""
Module to contain Machine Learning functions
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
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

    def temp_logistic_regression(self, x_data, y_data, dictionary) -> None:
        """
        Logistical regression function
        """
        # x_train, x_test, y_train, y_test = train_test_split(
        #    x_data, y_data, test_size=self.const["test_size"]
        # )
        # model = LogisticRegression()
        # model.fit(x_train, y_train)
        # predicted_x = model.predict(x_test)
        # fpr, tpr, thresholds = roc_curve(y_test, predicted_x)
        # score = auc(fpr, tpr)
        # print(score)

        for sub in x_data.keys():
            x_val = np.array(x_data[sub])
            y_val = np.array([[val] for val in y_data[sub]])
            logging.info("%s %s", len(x_val), len(y_val))
            counts = [sum(x_val[:, i]) for i in range(len(x_val[0]))]
            highest = max(counts)
            logging.info(
                "Most popular word: '%s' with %s occurances",
                dictionary[sub][counts.index(highest)],
                highest,
            )

    def temp_kneighbors(self, x_data, y_data) -> None:
        """
        KNeighbors function
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=self.const["test_size"]
        )
        model = KNeighborsClassifier(n_neighbors=self.const["K_val"])
        model.fit(x_train, y_train)
        score = cross_val_score(model, x_test, y_test, cv=self.const["cross_val"])
        logging.info(score)

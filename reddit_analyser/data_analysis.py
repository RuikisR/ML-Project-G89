from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
import yaml
import numpy as np


class DataAnalyser:
    def __init__(self):
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
        self.const = self.config["analyser_config"]

    def temp_logistic_regression(self, x_data, y_data, dictionary) -> None:
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
            x = np.array(x_data[sub])
            y = np.array([[val] for val in y_data[sub]])
            print(len(x), len(y))
            counts = [sum(x[:, i]) for i in range(len(x[0]))]
            highest = max(counts)
            print(
                f"Most popular word: {dictionary[sub][counts.index(highest)]} with {highest}"
            )

    def temp_kneighbors(self, x_data, y_data, dictionary) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=self.const["test_size"]
        )
        model = KNeighborsClassifier(n_neighbors=self.const["K_val"])
        model.fit(x_train, y_train)
        score = cross_val_score(model, x_test, y_test, cv=self.const["cross_val"])
        print(score)

import numpy as np
import pandas as pd
import util
from sklearn.metrics import accuracy_score, classification_report

def main():
    X_train, X_test, y_train, y_test = util.get_data(1)

    # features = ['horse_age', 'horse_rating', 'declared_weight', 'actual_weight', 'draw', 'horse_country', 'horse_type']

    X_params, y_params = util.nbayes_fit(X_train.values, y_train.values)

    y_pred = util.nbayes_predict(X_test.values, X_params, y_params)

    # money = util.gamble(X_test, y_pred, "win")

    # print("Net return: ", money)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
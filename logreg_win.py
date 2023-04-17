import util
import logreg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def main():

    X_train, X_test, y_train, y_test = util.get_data(1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = logreg.LogisticRegression(learning_rate=0.1, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    money = util.gamble(X_test, y_pred, y_test, "win")

    print("Net return: ", money)
    print("Accuracy:", accuracy_score(y_test['top_pos'], y_pred))

if __name__ == "__main__":
    main()
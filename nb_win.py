import util
import naivebayes
from sklearn.metrics import accuracy_score, classification_report

def main():
    X_train, X_test, y_train, y_test = util.get_data(1)

    clf = naivebayes.NaiveBayes(6)

    X_params, y_params = clf.fit(X_train.values, y_train.values)

    y_pred = clf.predict(X_test.values, X_params, y_params)

    util.gamble(y_pred, y_test, "win")

    print("Accuracy: ", accuracy_score(y_test['top_pos'], y_pred))
    # print(classification_report(y_test['top_pos'], y_pred))


if __name__ == "__main__":
    main()
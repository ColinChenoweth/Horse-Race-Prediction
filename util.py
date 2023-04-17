import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from imblearn.under_sampling import RandomUnderSampler


def get_data(top_pos):

    races = pd.read_csv('./data/races.csv')
    runs = pd.read_csv('./data/runs.csv')

    data = pd.merge(races, runs, on='race_id')

    data['top_pos'] = data['result'].apply(lambda x: 1 if x <= top_pos else 0)

    features = ['horse_age', 'horse_rating', 'declared_weight', 'actual_weight', 'win_odds', 'place_odds', 'draw', 'horse_country', 'horse_type']
    X = data[features]
    # y = data['top_pos']
    y = data

    X = pd.get_dummies(X, columns=['horse_country', 'horse_type'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27, stratify=y['top_pos'])

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)

    a = np.unique(y_train['top_pos'], return_counts=True)
    cur_ss = a[1][1]/(a[1][0])

    undersampler = RandomUnderSampler(sampling_strategy=cur_ss**(1/2))
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train['top_pos'])

    return X_train_resampled, X_test, y_train_resampled, y_test

def nbayes_fit(X, y):
    n_examples, n_features = X.shape

    y_labels, y_counts = np.unique(y, return_counts=True)

    # nd.array of dictionaries mapping values to the likelihood
    X_params = np.ndarray(shape=(len(y_labels), n_features), dtype=dict)
    # prior
    y_params = y_counts/n_examples

    # For every class label
    for i, y_label in enumerate(y_labels):
        # For every feature column
        for j, col in enumerate(X.T):
            if j < 6:
                mu = np.sum(col[y == y_label]) / y_counts[i]
                sigma = np.sqrt(np.sum((col[y == y_label] - mu)**2) / y_counts[i])
                thetas = np.array([mu, sigma])
                x_labels = np.array(["mu", "sigma"])
            else:
                x_labels, x_counts = np.unique(col[y == y_label], return_counts=True)
                # p = 1/len(np.unique(col))
                # m = 1/p
                # thetas = (x_counts+m*p)/(y_counts[i]+m)

                # MLE probability parameters with smoothing
                thetas = x_counts/y_counts[i]
                thetas = np.log(thetas)

            # Store a dictionary as a parameter map
            # Which maps from the feature value to the MLE probability parameter
            X_params[i][j] = dict(zip(x_labels, thetas))

    return X_params, y_params

def nbayes_predict(X, X_params, y_params):
    y_hat = np.empty(len(X))
    # For each example x, find the most probable hypothesis
    for i, x in enumerate(X):
        # For each hypothesis (class label), consider the likelihood that the hypothesis holds for x
        hypotheses = np.copy(y_params)
        for j in range(len(hypotheses)):
            # For each feature in x, consider the likelihood that the hypothesis holds for that feature
            for k, feature in enumerate(x):
                if k < 6:
                    hypotheses[j] += norm.pdf(x[k],loc=X_params[j][k]["mu"],scale=X_params[j][k]["sigma"])
                else:
                    try:
                        hypotheses[j] += X_params[j][k][feature]
                    except:
                        hypotheses[j] += 0
                        # print("feature not in training", k)
        # Pick the most probable hypothesis
        y_hat[i] = np.argmax(hypotheses)

    return y_hat

def gamble(X, y_pred, y_true, bet):
    bets = np.count_nonzero(y_pred)
    print("Num Bets: ", bets)
    money = -10*bets
    if bet == 'win':
        money += np.sum(10 * y_true['win_odds'].values[(y_pred == 1) & (y_true['top_pos'] == 1)])
    elif bet == 'place':
        money += np.sum(10*y_true['place_odds'].values[(y_pred == 1) & (y_true['top_pos'] == 1)])
    return money

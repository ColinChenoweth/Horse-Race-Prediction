import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


def get_data(top_pos):

    races = pd.read_csv('./data/races.csv')
    runs = pd.read_csv('./data/runs.csv')

    data = pd.merge(races, runs, on='race_id')

    data['top_pos'] = data['result'].apply(lambda x: 1 if x <= top_pos else 0)

    # make sure all continuous features come first, then draw and all other nominal features
    # also make sure to set/change num_cont_features correctly when creating NaiveBayes
    features = ['horse_age', 'horse_rating', 'declared_weight', 'actual_weight', 'win_odds', 'place_odds', 'draw', 'horse_country', 'horse_type']
    X = data[features]
    y = data

    X = pd.get_dummies(X, columns=['horse_country', 'horse_type'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27, stratify=y['top_pos'])

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)

    a = np.unique(y_train['top_pos'], return_counts=True)
    cur_ss = a[1][1]/(a[1][0])

    # sampling_strategy is the what percent of the majority the minority class will become
    # meaning if sampling_strategy = 0.5 then the minority class with have have as many examples as the majority classs
    undersampler = RandomUnderSampler(sampling_strategy=cur_ss**(1/2))
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train['top_pos'])

    # x_train_resampled, X_test only have the features defined in the set 
    # y_train_resampled only has feature top_pos
    # y_test has all features in the dataset
    return X_train_resampled, X_test, y_train_resampled, y_test

def gamble(X, y_pred, y_true, bet):
    bets = np.count_nonzero(y_pred)
    print("Num Bets: ", bets)
    money = -10*bets
    if bet == 'win':
        money += np.sum(10 * y_true['win_odds'].values[(y_pred == 1) & (y_true['top_pos'] == 1)])
    elif bet == 'place':
        money += np.sum(10*y_true['place_odds'].values[(y_pred == 1) & (y_true['top_pos'] == 1)])
    return money

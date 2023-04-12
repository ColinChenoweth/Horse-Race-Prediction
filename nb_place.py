import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

races = pd.read_csv('./data/races.csv')
runs = pd.read_csv('./data/runs.csv')

data = pd.merge(races, runs, on='race_id')

data['top3'] = data['result'].apply(lambda x: 1 if x <= 3 else 0)

features = ['horse_age', 'horse_rating', 'declared_weight', 'actual_weight', 'draw', 'horse_country', 'horse_type']
X = data[features]
y = data['top3']

X = pd.get_dummies(X, columns=['horse_country', 'horse_type'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
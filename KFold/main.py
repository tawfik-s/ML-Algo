# split using kfold and fit rf classifier  and prdict the output
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('./diabetics.csv')


X = data.drop('outcome', axis=1)
y = data['outcome']

rf = RandomForestClassifier(n_estimators=10)


k = 5
kfold = KFold(n_splits=k , random_state=None, shuffle=True)

acclist = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_test, predictions)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, predictions)
    print("accuracy score ", acc)
    acclist.append(acc)


acc = sum(acclist) / k

print(acc)

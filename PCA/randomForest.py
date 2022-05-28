import pandas as pd

data = pd.read_csv('pcadata.csv')
X = data.drop('outcome', 1)
y = data['outcome']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train)

# prediction
predictions = rf.predict(x_test)

# compare prediction with y_test
# confusion matrix
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, predictions)
print(matrix)

#               PREDICTED
#         tn.                  fp
# ACTUAL
#         fn                   tp

# accuracy score
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, predictions)
print("accuracy ", acc)

# precision
from sklearn.metrics import precision_score

pre = precision_score(y_test, predictions)
print("precision ", pre)

# recall
from sklearn.metrics import recall_score

rec = recall_score(y_test, predictions)
print("recall  ", rec)

# f1-measure
from sklearn.metrics import f1_score

f1 = f1_score(y_test, predictions)
print("f1-measure  ", f1)

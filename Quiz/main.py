# read the dataset
# divide it train test splet  60 to 40
# classfication using svm
# kernel linear and kerneal rbf
# acuuracy and presetion and f1 measure

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('./classficationData/classifier.csv')  # dataframe
X = df.drop('outcome', 1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
svc_model = SVC(C=.1, kernel='linear', gamma=1)
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, prediction)
print("accuracy ", acc)

# precision
from sklearn.metrics import precision_score

pre = precision_score(y_test, prediction)
print("precision ", pre)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, prediction)
print("f1-measure  ", f1)

print("--------------------kernel rbf-------------------------------")

svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, prediction)
print("accuracy ", acc)

# precision
from sklearn.metrics import precision_score

pre = precision_score(y_test, prediction)
print("precision ", pre)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, prediction)
print("f1-measure  ", f1)

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import datasets

# we will use iris data set

iris = datasets.load_iris()
x = iris.data
y = iris.target

# creating training and test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# feature scalling

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# Training A SVM classifier using SVC class

classifiere = SVC(kernel='rbf')

classifiere.fit(x_train_std, y_train)

# Mode performance

y_pred = classifiere.predict(x_test_std)
print('Accuracy ', accuracy_score(y_test, y_pred))



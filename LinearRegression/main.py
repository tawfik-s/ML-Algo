import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_x = diabetes_x[:, np.newaxis, 2]  # he take the second line

# split the data into training and testing groups

x_train, x_test, y_train, y_test = train_test_split(
    diabetes_x, diabetes_y, test_size=0.3)


regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

print("Coefficient \n", regr.coef_)
# the mean_squared_error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Plot outputs
plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

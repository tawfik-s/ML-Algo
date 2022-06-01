from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

IrisData = load_iris()

y = IrisData.target
x = IrisData.data

print(x[:5])
print(y[:5])
print(IrisData.feature_names)

from sklearn.datasets import load_digits;

DigitsData = load_digits()
print(DigitsData.feature_names)

plt.gray()
for g in range(10):
    print("images number ", g)
    plt.matshow(DigitsData.images[g])
    print("_______________________")
    plt.show()

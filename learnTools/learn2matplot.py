import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [5, 7, 4]
x2 = [1, 2, 3]
y2 = [10, 12, 14]
plt.plot(x, y, label="first line")
plt.bar(x2, y2, label="second line", color="r")
plt.xlabel("the x value")
plt.ylabel("the y value")
plt.title("my crazy graph")


plt.show()

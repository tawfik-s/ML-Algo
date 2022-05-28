import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('diabetics.csv')  # dataframe
X = df.drop('outcome', 1)
y = df['outcome']

pca = PCA(n_components=5)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data=principalComponents, columns=['A', 'B', 'C', 'D', 'E'])
finalDf = pd.concat([principalDf, y], axis=1)

finalDf.to_csv("pcadata.csv", index=False, header=True)  # save final df to a csv file

exvar = pca.explained_variance_ratio_
cexvarsum = np.cumsum(exvar)

print(exvar)
plt.bar(range(0, len(exvar)), exvar, label='Individual explained variance')

plt.step(range(0, len(cexvarsum)), cexvarsum, label='Cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')

plt.legend(loc='lower right')

plt.show()

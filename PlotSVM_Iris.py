#Plotting Support Vector Machine
'''
Pada plot SVM terdapat garis hyper plane yang berfungsi memisahkan data ke dalam masing-masing klasifikasinya.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#=============================================
#Load Data
#=============================================

iris = load_iris()
# print(iris)
# print(dir(iris))

#=============================================
#Create DataFrame
#=============================================

dfIris = pd.DataFrame(
    iris['data'],
    columns = iris['feature_names']
)
dfIris['target'] = iris['target']
dfIris['species'] = dfIris['target'].apply(
    lambda index: iris['target_names'][index]           #lambda = anonymous function
)
# print(dfIris.head())
# print(dfIris.tail())

#=============================================
#Import SVM Model
#=============================================

from sklearn.svm import SVC
model = SVC(gamma = 'auto')

#=============================================
#Separate DataFrame by Its Species
#=============================================

dfSetosa = dfIris[dfIris['target'] == 0 ]
# print(dfSetosa)
dfVersicolor = dfIris[dfIris['target'] == 1 ]
# print(dfVersicolor)
dfVirginica = dfIris[dfIris['target'] == 2 ]
# print(dfVirginica)

# ============================================
# Plot SVM Sepal
# ============================================

#Create hyper plane by meshgrid
def create_meshgrid(x, y):
    x_min = x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, .02),
        np.arange(y_min, y_max, .02)
    )
    return xx, yy

sepal = iris['data'][:, :2] #[:, :2] semua baris, mulai dari column 0 sampai sebelum 2
# print(sepal)

##x0: sepal length, x1: sepal width
x0, x1 = sepal[:, 0], sepal[:, 1]    #[:, 0] semua baris dari kolom ke-0 | [:, 1] semua baris dari kolom ke-1
xx, yy = create_meshgrid(x0, x1)
# print(xx)
# print(yy)

model.fit(sepal, iris['target'])

#Plot Data
def plotSVM(ax, model, xx, yy, **params):
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    hasil = ax.contourf(xx, yy, z, **params)
    return hasil

fig = plt.figure('SVM', figsize = (13,6))
ax = plt.subplot(121)
plotSVM(ax, model, xx, yy, cmap = 'rainbow', alpha = .2)
plt.scatter(
    dfSetosa['sepal length (cm)'],
    dfSetosa['sepal width (cm)'],
    color = 'r',
    marker = 'o',
    label = 'Setosa'
)

plt.scatter(
    dfVersicolor['sepal length (cm)'],
    dfVersicolor['sepal width (cm)'],
    color = 'y',
    marker = 'o',
    label = 'Versicolor'
)

plt.scatter(
    dfVirginica['sepal length (cm)'],
    dfVirginica['sepal width (cm)'],
    color = 'b',
    marker = 'o',
    label = 'Virginica'
)

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_title('Sepal length (cm) vs Sepal width (cm)')
ax.legend()
ax.grid(True)

# ============================================
# Plot SVM Petal
# ============================================

#Create hyper plane by meshgrid
def create_meshgrid(x, y):
    x_min = x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, .02),
        np.arange(y_min, y_max, .02)
    )
    return xx, yy

petal = iris['data'][:, 2:] #[:, 2:] semua baris, mulai dari column 2 sampai habis
print(petal)

##x0: petal length, x1: petal width
x0, x1 = petal[:, 0], petal[:, 1]    #[:, 0] semua baris dari kolom ke-0 | [:, 1] semua baris dari kolom ke-1
xx, yy = create_meshgrid(x0, x1)
# print(xx)
# print(yy)

model.fit(petal, iris['target'])

#Plot Data
def plotSVM(ax, model, xx, yy, **params):
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    hasil = ax.contourf(xx, yy, z, **params)
    return hasil

fig = plt.figure('SVM')
ax = plt.subplot(122)
plotSVM(ax, model, xx, yy, cmap = 'rainbow', alpha = .2)
plt.scatter(
    dfSetosa['petal length (cm)'],
    dfSetosa['petal width (cm)'],
    color = 'r',
    marker = 'o',
    label = 'Setosa'
)

plt.scatter(
    dfVersicolor['petal length (cm)'],
    dfVersicolor['petal width (cm)'],
    color = 'y',
    marker = 'o',
    label = 'Versicolor'
)

plt.scatter(
    dfVirginica['petal length (cm)'],
    dfVirginica['petal width (cm)'],
    color = 'b',
    marker = 'o',
    label = 'Virginica'
)

ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.set_title('Petal length (cm) vs Petal width (cm)')
ax.legend()
ax.grid(True)


plt.show()
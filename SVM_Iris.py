#ML Support Vector Machine
'''
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
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
    lambda index: iris['target_names'][index]          
)
# print(dfIris.head())
# print(dfIris.tail())

#=============================================
#Split Train (90%) and Test (10%)
#=============================================

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    dfIris[[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]],
    dfIris['species'],
    test_size = .1
)
# print(len(x_train))
# print(len(x_test))
# print(len(y_train))
# print(len(y_test))

#=============================================
#SVM Algorithm
#=============================================

#Import SVC
from sklearn.svm import SVC
model = SVC(gamma = 'auto')

#Training
model.fit(x_train, y_train)

#Prediction
# print(model.predict([[5.1, 3.5, 1.4, 0.2]]))     #x_train[0]
# print(model.predict([[7.0, 3.2, 4.7, 1.4]]))     #x_train[50]
# print(model.predict([[6.3, 3.3, 6.0, 2.5]]))     #x_train[100]
print(model.predict([x_test.iloc[0]]))
print(y_test.iloc[0])

#Accuracy
print(model.score(x_test, y_test) * 100, '%')
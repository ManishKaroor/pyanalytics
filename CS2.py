# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:23:57 2020

@author: Manish Karoor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns
df = data('mtcars')
df.head()
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
df['am'].value_counts()

#classification
#predict if transmission of car is 0 or 1 on basis of mpg, hp, wt
X1 = df[['mpg','hp','wt']]
Y1 = df['am']
Y1.value_counts()

#regression
#predict if mpg (numerical value) on basis of am, hp, wt
X2 = df[['am','hp','wt']]
Y2 = df[['mpg']]
Y2
np.mean(Y2)
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=.20)
X1_train.shape
X1_test.shape

from sklearn.tree import DecisionTreeClassifier
clsModel = DecisionTreeClassifier()  #model with parameter
clsModel.fit(X1_train, Y1_train)

ypred1 = clsModel.predict(X1_test)
len(ypred1)

ypred1
Y1_test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
classification_report(y_true=Y1_test, y_pred= ypred1)
confusion_matrix(y_true=Y1_test, y_pred=ypred1)
accuracy_score(y_true=Y1_test, y_pred=ypred1)
pd.DataFrame({'Actual': Y1_test, 'Predicted': ypred1})


X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=.20, random_state=123)
X2_train.shape
X2_test.shape


from sklearn.tree import DecisionTreeRegressor
regrModel = DecisionTreeRegressor()  
regrModel.fit(X2_train, Y2_train)
ypred2 = regrModel.predict(X2_test)
ypred2
Y2_test
metrics.mean_absolute_error(y_true=Y2_test, y_pred=ypred2)
metrics.mean_squared_error(y_true=Y2_test, y_pred=ypred2)
np.sqrt(metrics.mean_squared_error(y_true=Y2_test, y_pred=ypred2))
df1= Y1_test
df1['ypred2']=ypred2
df1

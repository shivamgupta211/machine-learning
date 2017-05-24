import pandas as pd
import math
import numpy as np
from sklearn import preprocessing,model_selection, neighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)

df.drop(['id'],1,inplace=True)


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.1)

clf = neighbors.KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
clf.fit(X_train,y_train)

accu = clf.score(X_test,y_test)
print(accu)

ex = np.array([[4,2,1,1,1,2,3,2,1]])
ex = ex.reshape(len(ex),-1)
pred = clf.predict(ex)

print(pred)

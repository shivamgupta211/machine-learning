import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation,model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = pd.read_csv('quandl_csv_google.csv')

df['HL_PCT'] = (df['Adj. Open']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

df.fillna(-9999,inplace=True)
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'],1))


X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


#print(X[:-forecast_out])

df.dropna(inplace=True)


y = np.array(df['label'])
print(y[-forecast_out:])

X_train, X_test,y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1)

clf = LinearRegression()
clf.fit(X_train,y_train)
'''
with open('lr.pickle','wb') as F:
    pickle.dump(clf,F)'''

pickle_in = open('lr.pickle','rb')
clf = pickle.load(pickle_in)

accu = clf.score(X_test,y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set, accu, forecast_out)

df['forecast'] = np.nan

df['label'].plot()
df['forecast'].plot()
plt.show()

















print(accu)

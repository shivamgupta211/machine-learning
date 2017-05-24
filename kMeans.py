import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.cluster import KMeans


df = pd.read_excel('titanic.xls');
df.drop(['name','body'],1,inplace=True)
df.fillna(0,inplace=True)

#   converts the string fields to the int fields with value ranging from 0 to len(set(column))
for col in df.columns:
    #print(df[col].values)
    if df[col].dtypes != 'int64' and df[col].dtypes != 'float64':
        uniqueValue = set(df[col])
        dic = {}
        x=0
        for uVal in uniqueValue:
            dic[uVal] = x
            x = x+1 
        df[col].replace(to_replace=dic,inplace=True)
        
             
    
print(df.head())

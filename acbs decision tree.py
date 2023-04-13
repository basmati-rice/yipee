#Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
df=pd.read_csv(r"C:\Users\HP\Downloads\heart.csv")#Reading the data
x=df.values[:,:13]
print()
print(x.shape)
y=df.values[:,12:13]
print()
print(y)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
clf_entropy=DecisionTreeClassifier()
clf_entropy.fit(x_train,y_train)
y_pred_en=clf_entropy.predict(x_test)
print(y_pred_en)
print("Accuracy is:",accuracy_score(y_test,y_pred_en)*100)

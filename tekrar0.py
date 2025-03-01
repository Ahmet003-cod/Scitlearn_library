# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:39:25 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
iris=load_iris()
print(iris)
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df["target"]=iris.target
print(),print()
#print(df)
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)

std=StandardScaler()
X_train=std.fit_transform(X_train)
X_test=std.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("doğruluk orani=",accuracy)

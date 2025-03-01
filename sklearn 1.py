# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:37:18 2025
@author: Huzur Bilgisayar
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer#içindeki seten indirme algoritması
from sklearn.neighbors import KNeighborsClassifier#öğrenim algoritmasi
from sklearn.metrics import accuracy_score,confusion_matrix #doğruluk scoruna bakıyor
from sklearn.model_selection import train_test_split#test süreci için libray
from sklearn.preprocessing import StandardScaler
kanser=load_breast_cancer()
df=pd.DataFrame(data=kanser.data,columns=kanser.feature_names)
df["target"]=kanser.target
print(df) 
print(type(kanser))
X=kanser.data#(features)(özellikler)
y=kanser.target#(sonuc)
#train test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.50,random_state=42)
#ölçeklendirme
sdc=StandardScaler()
X_train=sdc.fit_transform(X_train)
X_test=sdc.transform(X_test)

#knn modeli oluştur ve train et
knn=KNeighborsClassifier(n_neighbors=3)# Myodel oluştuma komşu prametresini unutma(sınıflandırıcı)
knn.fit(X_train,y_train)#verimizi kullanarak knn algoritmamızı eğitir
y_pred=knn.predict(X_test)#model eğitildi ve bizi tahmin veriyor

print(y_pred)#tahminleri
print()
print(y)#doğru sonuclar
print(),print()
accuarcy=accuracy_score(y_test,y_pred)#burda doğrulu % bize verecek
conf_matrix=confusion_matrix(y_test,y_pred)#Bize net bigi verce 0 ve 1lerden kaç doğru kaç yanlışımız olduğuna dair.
print("doğruluk %=",accuarcy)#doğruluk %=0.9590643274853801
print("confusion matrix"),print(conf_matrix)


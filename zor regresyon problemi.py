# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:59:16 2025
@author: Huzur Bilgisayar
"""
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.neighbors import KNeighborsRegressor
x=np.sort(10*np.random.rand(40,1),axis=0)#uniform 1 ile 5 arsında 40 sayi#küçükten büyüğe
print(x)

y=np.sin(x).ravel()#target#ravel fonksiyonu
print(y)

plt.scatter(x,y,linewidth=1,marker="o",linestyle="--")

#add nois
y[::5]+=1*(0.5-np.random.rand(8))
plt.scatter(x, y)

T=np.linspace(0,10,500)[:,np.newaxis]#1 boyutlu veriyi newaxis sayesinde iki boyutlu veri olur
for i,weight in enumerate(["uniform","distance"]):#for dögüsü sayesinde hem uniform hem de distance alıyoruz
    knn=KNeighborsRegressor(n_neighbors=5,weights=weight)
    y_pred=knn.fit(x, y).predict(T)#hem eğitim hem tahmin
    plt.subplot(2,1,i+1)#2 yatay grafik 1 sütün grafik ve nerden alsın 1.graif uniform 2. grafik distance
    plt.scatter(x, y,color="red",label="data")
    plt.scatter(T,y_pred,color="blue",label="prediction")
    plt.axis("tight")#verinin etrafında gereksiz boşluk bırakmadan çizim yapar.
    plt.legend()#label çalışması için
    plt.title("KNn regresyon weight={}".format(weight))
plt.tight_layout()#axis fonk çalışması için
plt.show()



#bir eğitici örnek

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)



functions = [np.sin, np.cos]

plt.figure(figsize=(6,6))  

for i, func in enumerate(functions):  
    plt.subplot(2, 1, i+1)  # i=0 → 1. grafik, i=1 → 2. grafik
    plt.plot(x, func(x),color="red")
    plt.title(f"{func.__name__} Grafiği")  

plt.tight_layout()
plt.show()



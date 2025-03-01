# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:27:03 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot  as plt


#veri seti tanımlama
iris=load_iris()
X=iris.data#features
y=iris.target #target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#DI modelini oluştur ve train et
tree_clf=DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)#saflık olarak bölmeye çalışır
tree_clf.fit(X_train,y_train)#max deepth=ne kadar veri ağaç dalı olmasını gösteriri
#Dı evaluation test
y_pred=tree_clf.predict(X_test)
accuary=accuracy_score(y_test, y_pred)
print("iris eğitim seti ile DI veri modelin doğrulığu=",accuary)#%100
conf_matrix=confusion_matrix(y_test, y_pred)
print("confusion_matrix=",conf_matrix)
#veri göreleştirmesini(soy ağacı gibi)
plt.figure(figsize=(15,10))#figsize =yakınlı uzaklık durumu
plot_tree(tree_clf,filled=True,feature_names=iris.feature_names,class_names=list(iris.target_names))
plt.show()
feature_importens=tree_clf.feature_importances_#en önemli olanı bize gösteriyor
feature_names=iris.feature_names
feature_importend_sorted=sorted(zip(feature_importens,feature_names),reverse=True)
for importance,feature_name in feature_importend_sorted:
    print(f"{feature_name}:{importance}")



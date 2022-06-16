import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

credit_data=pd.read_csv("credit_data.csv")
#input paramaters multiple logistic regression
features=credit_data[["income","age","loan"]]
target=credit_data.default
#transforms dataframes into array
x=np.array(features).reshape(-1,3)
y=np.array(target)
#minmax transformation with preprocessing distance between [0.1]
x=preprocessing.MinMaxScaler().fit_transform(x)
feature_train,feature_test,target_train,target_test=train_test_split(x,y,train_size=0.3)
#the algorithim is going to use the 20 nearest neighbours to know how to classify the given items
classifier=KNeighborsClassifier(n_neighbors=32)
fitted_model=classifier.fit(feature_train,target_train)
predictions=fitted_model.predict(feature_test)
cross_valid_scores=[]
for k in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    cross_valid_scores.append(scores.mean())
print("optimal k with cross-validtion is",np.argmax(cross_valid_scores))
print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
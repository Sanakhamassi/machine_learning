import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
#mesure the accuracy with confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

credit_data=pd.read_csv("credit_data.csv")

features=credit_data[["income","age","loan"]]
target=credit_data.default
feature_train,feature_test,target_train,target_test=train_test_split(features,target,train_size=0.3)

model=GaussianNB()
fittedmodel=model.fit(feature_train,target_train)
predictions=fittedmodel.predict(feature_test)
print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
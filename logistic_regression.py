import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

credit_data=pd.read_csv("credit_data.csv")
#input paramaters multiple logistic regression
features=credit_data[["income","age","loan"]]
target=credit_data.default
#test_size=0.3 means that 30% data-set is for testing and 70% data is for training
feature_train,feature_test,target_train,target_test=train_test_split(features,target,train_size=0.3)
model=LogisticRegression()
model.fit=model.fit(feature_train,target_train)
#after the fit function been executed with the gradiant decent we will be able to estimate the bvalue
prediction=model.fit.predict(feature_test)
print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))
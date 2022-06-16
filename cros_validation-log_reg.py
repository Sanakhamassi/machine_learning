import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
credit_data=pd.read_csv("credit_data.csv")
#input paramaters multiple logistic regression
features=credit_data[["income","age","loan"]]
target=credit_data.default
#transforms dataframes into array
x=np.array(features).reshape(-1,3)
y=np.array(target)
model=LogisticRegression()
#cv=5 we splited into 5  folks the defaut value is 3
#cross validate returns a dictionnary that contains features that i can acess using th key
predicted=cross_validate(model,x,y,cv=5)
#shows the accuracy ofthe algorithm
print(predicted['test_score'])
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
#read data into a dataframe
houses_data=pd.read_csv("house_prices.csv")
size=houses_data['sqft_living']
price=houses_data['price']
#machine learning handle arrays not data frames
#reshape to get rid of index column reshape(-1)
x=np.array(size).reshape(-1,1)
y=np.array(price).reshape(-1,1)

#we use linear regression to find the linear relation between the size and the price h(x)=b0x+b1
model=LinearRegression()
#when we call fit sklearn will train the model with gradient-decent
model.fit(x,y)
#MSE value andR value
regression_model_mse=mean_squared_error(x,y)
print("MSE:",math.sqrt(regression_model_mse))
#should be close to one to confirm the linear regression relation
print("R sqaured value:",model.score(x,y))
#we can get the b values after the model fit
#b1
print(model.coef_[0])
print(model.intercept_[0])
#visualise the data set after fitted

plt.scatter(x,y,color="green")
plt.plot(x,model.predict(x),color='black')
plt.title("linear regression")
plt.xlabel("size")
plt.ylabel("Price")
plt.show()

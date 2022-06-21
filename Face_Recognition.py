from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti_data=fetch_olivetti_faces()
#there are 400images-10*40(40 persons - each 1 has 10 images)1 image=64*64 pixels
features=olivetti_data.data
#we represents target values(people) with faceid
target=olivetti_data.target

"""fig,subplot=plt.subplots(nrows=5,ncols=8,figsize=(14,8))
#in one array
subplot=subplot.flatten()
for unique_user_id in np.unique(target):
    image_index=unique_user_id*8
    subplot[unique_user_id].imshow(features[image_index].reshape(64,64),cmap='gray')
    subplot[unique_user_id].set_xticks([])
    subplot[unique_user_id].set_yticks([])
    subplot[unique_user_id].set_title("Face id : %s"%unique_user_id)
plt.suptitle("the dataset of 40 people")
plt.show()
fig,subplot=plt.subplots(nrows=1,ncols=10,figsize=(18,9))
for j in range(10):
    subplot[j].imshow(features[j].reshape(64,64),cmap='gray')
    subplot[j].set_xticks([])
    subplot[j].set_yticks([])
plt.show()"""
#split the original dataset into traning+test data
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.25,stratify=target,random_state=0)
#let's try to find the eginvectors(principal components
pca=PCA(n_components=100,whiten=True)
pca.fit(x_train)
x_pca=pca.fit_transform(features)
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)
print(features.data)
models=[("Logistic regression",LogisticRegression()),("Support vector machine",SVC()),("Naive baise classifier",GaussianNB())]
for name,model in models:
    kfold=KFold(n_splits=5,shuffle=True,random_state=0)
    cv_scores=cross_val_score(model,x_pca,target,cv=kfold )
    print("Mean of the cross validation scores:%s"% cv_scores.mean())




from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

digits=load_digits()

x_digits=digits.data
#one dimentionnel array
y_digits=digits.target
#trsfrom the 64 features into 10 features
estimator=PCA(n_components=10)
x_pca=estimator.fit_transform(x_digits)
colors=['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
for i in range(len(colors)):
    px=x_pca[:,0][y_digits==i]
    py=x_pca[:,1][y_digits==i]
    plt.scatter(px,py,c=colors[i])
    plt.legend(digits.target_names)

plt.xlabel("first principal component")
plt.ylabel("second principal component")
plt.show()


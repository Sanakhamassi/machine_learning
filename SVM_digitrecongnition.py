from sklearn import svm,metrics,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#we have images as the features and the target as the target value
digits=datasets.load_digits()
images_and_labels=list(zip(digits.images,digits.target))

"""for index ,(image,label)in enumerate(images_and_labels[:6]):
    plt.subplot(2,3,index+1)
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Target: %i'% label)
plt.show()"""
#to apply a classifier on this data we need a flatten the image: instead of a 8*8 matrix
#we have to use a one_dimensionnel array with 64 items
data=digits.images.reshape((len(digits.images),-1))
classifier=svm.SVC(gamma=0.001)
train_test_split=int(len(digits.images)*0.75)
classifier.fit(data[:train_test_split],digits.target[:train_test_split])

expected=digits.target[train_test_split:]
predicted=classifier.predict(data[train_test_split:])

print(metrics.confusion_matrix(expected,predicted))
print(accuracy_score(expected,predicted))

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for first iage",classifier.predict(data[-1].reshape(1,-1)))
plt.show()

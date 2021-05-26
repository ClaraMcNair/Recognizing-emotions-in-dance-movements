
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt   

#DATA
filename = 'MOVINGdata.csv'
names = ['speed','aceleration','openness','x','y','a','b','c','d','e','f','g']
#names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
#names = ['speed','aceleration','openness','x','y'] 
data = pandas.read_csv(filename, names=names)

# TARGET:
filename2 = 'MOVINGtarget.csv'
names2 = ['class']
target = pandas.read_csv(filename2, names=names2)

#reassign data and target to X and Y for simplicity
X = data
Y = target

##print(X.shape)
#print(Y.shape)

# we can then create the classifier with given data and target
#clf = DecisionTreeClassifier(random_state=0)
#clf = KNeighborsClassifier(n_neighbors=1)
clf = RandomForestClassifier(random_state=0)
'''
#testing prediction on random element from datalist
print(clf.predict([[0.08742308834540400,0]]))

#probabilty of each class
print(clf.predict_proba([[0.08742308834540400,0]]))
'''

#SPLITTING THE DATA
#Using built-in-function 'train_test_split' from scikit
#test_size 0.2 means that 20% will go to testing, the remaining 80% is for training

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#printing size of the two sets
#print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)


#Fitting the model using the training-set
clf.fit(X_train, Y_train)


#printing the features with highest weight
print('feature weight: ', clf.feature_importances_)
imortance = clf.feature_importances_

#Testing the model on the full testing-set
#comparing the predicted class with actual class

pred = clf.predict(X_test)
true = Y_test


#Showing the models accuracy(in percent) using built-in-function 'score'
# The test-set are given as arguments, that is the predicted class of X and the actual class, Y. 
print('Accuracy: ', clf.score(X_test,Y_test))

print(confusion_matrix(Y_test, clf.predict(X_test)))

#plot_confusion_matrix(clf, X_test, Y_test, cmap=plt.cm.Blues)  
#plt.show() 
feature_names = np.array(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12'])
plt.bar(feature_names, imortance)
plt.show()
# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNC
# print dataset
gl = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\KNN\\glass.csv")
gl.head()
g=gl.columns
# split datadset
X = gl.iloc[:, :-1]
y = gl.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# use scaler function
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# apply KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# line plot through function
glass1 = []
for i in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    pred_i = classifier.predict(X_test)
    glass1.append(np.mean(pred_i != y_test))
plt.plot(range(1, 100), glass1, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)

train_acc = np.mean(classifier.predict(X_train)==y_train)
test_acc = np.mean(classifier.predict(X_test)==y_test)

acc = []
for i in range(5,100,5):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    train_acc = np.mean(classifier.predict(X_train)==y_train)
    test_acc = np.mean(classifier.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
plt.plot(np.arange(5,100,5),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(5,100,5),[i[1] for i in acc],"ro-")

plt.plot(range(5,100,5), acc, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)







































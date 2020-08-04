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
z1 = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\1 SC\\KNN\\zoo.csv")
z1.head()
z2=z1.columns
# split dataset into train & test dataset
X = z1.iloc[:,1:-1]
y = z1.iloc[:, 17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# print ypred & fit values
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
z3 = KNeighborsClassifier(n_neighbors=5)
z3.fit(X_train, y_train)
y_pred = z3.predict(X_test)
# print confussion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# line plot through function
zoo1 = []
for i in range(1, 80):
    z3 = KNeighborsClassifier(n_neighbors=i)
    z3.fit(X_train, y_train)
    pred_i = z3.predict(X_test)
    zoo1.append(np.mean(pred_i != y_test))
plt.plot(range(1, 80), zoo1, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
# print meean values
train_acc = np.mean(z3.predict(X_train)==y_train)
test_acc = np.mean(z3.predict(X_test)==y_test)
# line plot through function
acc = []
for i in range(1,80,5):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    train_acc = np.mean(classifier.predict(X_train)==y_train)
    test_acc = np.mean(classifier.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
plt.plot(np.arange(1,80,5),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(1,80,5),[i[1] for i in acc],"ro-")

plt.plot(range(1,80,5), acc, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
















































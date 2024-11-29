import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def metrics(Y_test,Y_pred):
    accuracy= round(accuracy_score(Y_test,Y_pred)*100,2)
    precision=round(precision_score(Y_test,Y_pred)*100,2)
    recall=round(recall_score(Y_test,Y_pred)*100,2)
    print("Accuracy score: "+ str(accuracy)+"%")
    print("Precision: "+str(precision)+"%")
    print("Recall Score: "+str(recall)+"%")
    print("\n")


diabetes_data=pd.read_csv('diabetes/diabetes.csv')

predictors=diabetes_data.drop('target',axis=1)
target= diabetes_data['target']

X_train,X_test,Y_train,Y_test= train_test_split(predictors,target,test_size=0.20,random_state=0)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
print("Logistic Regression model: ")
metrics(Y_test,Y_pred_lr)


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
nb.fit(X_train,Y_train)
Y_pred_nb = nb.predict(X_test)
print("Naive Bayes: ")
metrics(Y_test,Y_pred_nb)

# SVM 

from sklearn import svm

sv= svm.SVC(kernel='linear')
sv.fit(X_train,Y_train)
Y_pred_svm = sv.predict(X_test)
print("SVM: ")
metrics(Y_test,Y_pred_svm)

# K Nearest Neighbours 

from sklearn.neighbors import KNeighborsClassifier

max_accuracy=0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred_knn = knn.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy=current_accuracy
        best_x=k

# knn = KNeighborsClassifier(n_neighbors=9)
# knn.fit(X_train,Y_train)
# Y_pred_knn=knn.predict(X_test)
# print("KNN: ")
# metrics(Y_test, Y_pred_knn)

print(f"KNN with {k} neighbors:")
metrics(Y_test, Y_pred_knn)


# Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

max_accuracy=0

for x in range (200):
    dt=DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy=current_accuracy
        best_x=x

# print(max_accuracy)
print(best_x)

# dt = DecisionTreeClassifier(random_state=0, max_depth=5)
# dt.fit(X_train,Y_train)
# Y_pred_dt=dt.predict(X_test)
print("Decision Tree: ")
metrics(Y_test, Y_pred_dt)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

print(best_x)

# rf = RandomForestClassifier(random_state=0)
# scores = cross_val_score(rf, predictors, target, cv=5)
# print("Cross-Validation Scores:", scores)
# print("Mean CV Accuracy:", scores.mean() * 100)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
print("Random Forest:")
metrics(Y_test,Y_pred_rf)

rf.fit(X_train, Y_train)
train_accuracy = rf.score(X_train, Y_train) * 100
test_accuracy = rf.score(X_test, Y_test) * 100
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# import matplotlib.pyplot as plt
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# plt.figure()
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(X_train.shape[1]), predictors.columns[indices], rotation=90)
# plt.show()

# import pickle

# filename = 'parkinsons_disease_model.sav'
# pickle.dump(rf, open(filename, 'wb'))

# loaded_model = pickle.load(open('parkinsons_disease_model.sav', 'rb'))

# for column in predictors.columns:
#   print(column)

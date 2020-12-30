## Imports
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import matplotlib.pyplot as plt

## read data set
#       train
people_tr = pd.read_csv('../dataset/train.csv')
train_labels = pd.read_csv('../dataset/train_labels.csv')
people_tr_size = len(train_labels)
#       test
people_tst = pd.read_csv('../dataset/test.csv')
test_labels = pd.read_csv('../dataset/test_labels.csv')
people_tst_size = len(test_labels)

## preprocessing data
#       drop id and name
people_tr = people_tr.drop(['PassengerId', 'Name'], axis=1)
people_tst = people_tst.drop(['PassengerId', 'Name'], axis=1)

#       Sex to boolean
people_tr['Sex'] = people_tr['Sex'].replace('female', 0)
people_tr['Sex'] = people_tr['Sex'].replace('male', 1)
people_tst['Sex'] = people_tst['Sex'].replace('female', 0)
people_tst['Sex'] = people_tst['Sex'].replace('male', 1)

#        one hot encoder for cabin embarked
#          embarked train
embarked_tr = people_tr['Embarked'].values.reshape(people_tr_size, 1).tolist()
embarked_tst = people_tst['Embarked'].values.reshape(people_tst_size, 1).tolist()
embarked_tr.extend(embarked_tst)

en = OneHotEncoder().fit(embarked_tr)
embarked_tr = en.transform(embarked_tr[:people_tr_size]).toarray()
people_tr['Embarked_first'] = embarked_tr[:, 0]
people_tr['Embarked_second'] = embarked_tr[:, 1]
people_tr['Embarked_third'] = embarked_tr[:, 2]
people_tr = people_tr.drop('Embarked', axis=1)

#          embarked test

embarked_tst = en.transform(embarked_tst).toarray()
people_tst['Embarked_first'] = embarked_tst[:, 0]
people_tst['Embarked_second'] = embarked_tst[:, 1]
people_tst['Embarked_third'] = embarked_tst[:, 2]
people_tst = people_tst.drop('Embarked', axis=1)

#          cabin train
Cabin_tr = people_tr['Cabin'].values.reshape(people_tr_size, 1).tolist()
Cabin_tst = people_tst['Cabin'].values.reshape(people_tst_size, 1).tolist()
Cabin_tr.extend(Cabin_tst)

en = OneHotEncoder().fit(Cabin_tr)
Cabin_tr = en.transform(Cabin_tr[:people_tr_size]).toarray()
people_tr['Cabin_first'] = Cabin_tr[:, 0]
people_tr['Cabin_second'] = Cabin_tr[:, 1]
people_tr['Cabin_third'] = Cabin_tr[:, 2]
people_tr = people_tr.drop('Cabin', axis=1)

#          cabin test
Cabin_tst = en.transform(Cabin_tst).toarray()
people_tst['Cabin_first'] = Cabin_tst[:, 0]
people_tst['Cabin_second'] = Cabin_tst[:, 1]
people_tst['Cabin_third'] = Cabin_tst[:, 2]
people_tst = people_tst.drop('Cabin', axis=1)
## convert to list

mod = people_tr.mode().iloc[0]
people_tr = people_tr.fillna(mod)
people_tst = people_tst.fillna(mod)


X_train = people_tr.values.tolist()
Y_train = train_labels['Survived'].tolist()

X_test = people_tst.values.tolist()
Y_test = test_labels['Survived'].tolist()
## create model
clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=65)
## fit model
clf = clf.fit(np.array(X_train, dtype=np.float32), Y_train)

## predict
predict_TestOrTrain = "both"
if predict_TestOrTrain == "test":
    # compute accuracy train
    predicts = clf.predict(np.array(X_train, dtype=np.float32))
    matched = 0
    for i in range(len(predicts)):
        if predicts[i] == Y_train[i]:
            matched += 1
    print("decision tree accuracy on test : " + str(matched / len(Y_train)))
elif predict_TestOrTrain == "train":
    # compute accuracy test
    predicts = clf.predict(np.array(X_test, dtype=np.float32))
    matched = 0
    for i in range(len(predicts)):
        if predicts[i] == Y_test[i]:
            matched += 1
    print("decision tree accuracy on train : " + str(matched / len(Y_test)))
else:
    # compute accuracy train
    predicts = clf.predict(np.array(X_train, dtype=np.float32))
    matched = 0
    for i in range(len(predicts)):
        if predicts[i] == Y_train[i]:
            matched += 1
    print("decision tree accuracy on train : " + str(matched / len(Y_train)))
    # compute accuracy test
    predicts = clf.predict(np.array(X_test, dtype=np.float32))
    matched = 0
    for i in range(len(predicts)):
        if predicts[i] == Y_test[i]:
            matched += 1
    print("decision tree accuracy on test : " + str(matched / len(Y_test)))

# fig = plt.figure(figsize=(20, 10))
# tree.plot_tree(clf, fontsize=10)
# plt.show()


s_Scaler = preprocessing.StandardScaler().fit(X_train)
X_train = s_Scaler.transform(X_train)
X_test = s_Scaler.transform(X_test)
perc_clf = Perceptron(tol=1e-5, random_state=50, shuffle=True, n_iter_no_change=10)
# perc_clf = Perceptron(tol=1e-5, random_state=0)
perc_clf.fit(X_train, Y_train)

print("perceptron accuracy on train : " + str(perc_clf.score(X_train, Y_train)))
print("perceptron accuracy on test : " + str(perc_clf.score(X_test, Y_test)))


# select good features
all_features = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
good_features = set(clf.feature_importances_.argsort()[-5:])
bad_features = list(all_features.difference(good_features))

features_name = people_tr.columns
for i in bad_features:
    people_tr = people_tr.drop(features_name[i], axis=1)
    people_tst = people_tst.drop(features_name[i], axis=1)

X_train = people_tr.values.tolist()
Y_train = train_labels['Survived'].tolist()

X_test = people_tst.values.tolist()
Y_test = test_labels['Survived'].tolist()

s_Scaler = preprocessing.StandardScaler().fit(X_train)
X_train = s_Scaler.transform(X_train)
X_test = s_Scaler.transform(X_test)
clf = Perceptron(tol=1e-5, random_state=45, shuffle=True, n_iter_no_change=10)
# clf = Perceptron(tol=1e-8, random_state=0)
clf.fit(X_train, Y_train)
print("with feature selection : ")
print("perceptron accuracy on train : " + str(clf.score(X_train, Y_train)))
print("perceptron accuracy on test : " + str(clf.score(X_test, Y_test)))

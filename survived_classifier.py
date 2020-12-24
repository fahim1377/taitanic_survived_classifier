## Imports
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

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
people_tr = people_tr.drop(['PassengerId','Name'],axis=1)
people_tst = people_tst.drop(['PassengerId','Name'],axis=1)

#       Sex to boolean
people_tr['Sex'] = people_tr['Sex'].replace('female',0)
people_tr['Sex'] = people_tr['Sex'].replace('male',1)
people_tst['Sex'] = people_tst['Sex'].replace('female',0)
people_tst['Sex'] = people_tst['Sex'].replace('male',1)

#        one hot encoder for cabin embarked
#           train
embarked_tr = people_tr['Embarked'].values.reshape(people_tr_size,1).tolist()
en = OneHotEncoder().fit(embarked_tr)
embarked_tr = en.transform(embarked_tr).toarray()
people_tr['Embarked_first'] = embarked_tr[:,0]
people_tr['Embarked_second'] = embarked_tr[:,1]
people_tr['Embarked_third'] = embarked_tr[:,2]
people_tr = people_tr.drop('Embarked',axis=1)

#           test
embarked_tst = people_tst['Embarked'].values.reshape(people_tst_size,1).tolist()

embarked_tst = en.transform(embarked_tst).toarray()
people_tst['Embarked_first'] = embarked_tst[:,0]
people_tst['Embarked_second'] = embarked_tst[:,1]
people_tst['Embarked_third'] = embarked_tst[:,2]
people_tst = people_tst.drop('Embarked',axis=1)


Cabin_tr = people_tr['Cabin'].values.reshape(people_tr_size,1).tolist()
en = OneHotEncoder().fit(Cabin_tr)
Cabin_tr = en.transform(Cabin_tr).toarray()
people_tr['Cabin_first'] = Cabin_tr[:,0]
people_tr['Cabin_second'] = Cabin_tr[:,1]
people_tr['Cabin_third'] = Cabin_tr[:,2]
people_tr = people_tr.drop('Cabin',axis=1)

Cabin_tst = people_tst['Cabin'].values.reshape(people_tst_size,1).tolist()

Cabin_tst = en.transform(Cabin_tst).toarray()
people_tst['Cabin_first'] = Cabin_tst[:,0]
people_tst['Cabin_second'] = Cabin_tst[:,1]
people_tst['Cabin_third'] = Cabin_tst[:,2]
people_tst = people_tst.drop('Cabin',axis=1)

X_train = people_tr.values.tolist()
Y_train = train_labels['Survived'].tolist()

X_test = people_tst.values.tolist()
Y_test = test_labels['Survived'].tolist()

## create model
clf = tree.DecisionTreeClassifier()

## fit model
clf = clf.fit(X_train,Y_train,check_input=False)

## predict
clf.predict(X_test)


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.use_inf_as_na = True


label_encoder = LabelEncoder()

#replacing missing values with mean
imp = Imputer(missing_values="NaN", strategy = 'mean', axis=0)

data = pd.read_csv('train.csv', header = 0)

#removing features that we don't need
del data["Ticket"]
del data["Cabin"]
del data["Name"]
del data["PassengerId"]


#data.fillna(data.mean())

labels = data["Survived"].values.tolist()
del data["Survived"]

data["Sex"] = data["Sex"].replace(['male'], 1)
data["Sex"] = data["Sex"].replace(['female'], 0)
data["Embarked"] = data["Embarked"].replace(['S'], 0)
data["Embarked"] = data["Embarked"].replace(['Q'], 1)
data["Embarked"] = data["Embarked"].replace(['C'], 2)


#print(data.head())

pclass = data["Pclass"].tolist()
pclass = np.array(pclass)
#converting python dataframe to list of lists
features = data.values.tolist()
features = imp.fit_transform(features)

#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import *
#clf = linear_model.LogisticRegression()
clf = RandomForestClassifier(n_estimators=29)
#clf = KNeighborsClassifier(n_neighbors=35)
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()
#x_test,x_train,y_test,y_train = train_test_split(features, labels, test_size = 0.25)
#clf.fit(x_train, y_train)

clf.fit(features, labels)
##limits = range(1, 31)
#scores = []

'''for i in limits:
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    #clf.acc(y_test, pred)
    print("accuracy: ", accuracy_score(y_test, pred))
    scores.append(accuracy_score(y_test,pred))

#preds = clf.predict(y_test)
plt.plot(limits, scores)
plt.show()
#print("Acurracy: ", accuracy_score(y_test, preds))
'''
#need to OneHotEncode Pclass & Embarked columns
print(data.head())


#reading input and testing now
#===========================TESTING================================================================================

data = pd.read_csv('test.csv', header = 0)

d = data["PassengerId"]

del data["Ticket"]
del data["Cabin"]
del data["Name"]

#writing data to a file
#first creating a dataframe

d  = data["PassengerId"].values.tolist()
del data["PassengerId"]

#replacing values of non numbers to numerical forms

data["Sex"] = data["Sex"].replace(['male'], 1)
data["Sex"] = data["Sex"].replace(['female'], 0)
data["Embarked"] = data["Embarked"].replace(['S'], 0)
data["Embarked"] = data["Embarked"].replace(['Q'], 1)
data["Embarked"] = data["Embarked"].replace(['C'], 2)

#converting python dataframe to list of lists
features = data.values.tolist()
features = imp.fit_transform(features)
result = clf.predict(features)
result = result.tolist()
df = pd.DataFrame({'PassengerId': d, 'Survived':result})
df.set_index('PassengerId', inplace=True)

df.to_csv('out.csv', sep=',')
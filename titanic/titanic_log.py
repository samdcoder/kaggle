from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.use_inf_as_na = True

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

#OneHotEncode
data = pd.get_dummies(data, columns=['Pclass', 'Embarked'])

features = data.values.tolist()
features = imp.fit_transform(features)



#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25    )


#clf = linear_model.LogisticRegression()
clf = RandomForestClassifier(n_estimators=20)
#clf = KNeighborsClassifier(n_neighbors=35)
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)
print(data.head())
print("accuracy with new method", clf.score(x_test,y_test))
#clf.fit(features, labels)
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

#OneHotEncode
data = pd.get_dummies(data, columns=['Pclass', 'Embarked'])

#converting python dataframe to list of lists
features = data.values.tolist()
features = imp.fit_transform(features)
result = clf.predict(features)
result = result.tolist()
df = pd.DataFrame({'PassengerId': d, 'Survived':result})
df.set_index('PassengerId', inplace=True)
df.to_csv('out.csv', sep=',')

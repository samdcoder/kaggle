from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


#converting python dataframe to list of lists
features = data.values.tolist()
features = imp.fit_transform(features)

#splitting the dataset
#x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


#classifier svm
clf = svm.SVC(gamma=0.0001, C=100)
clf.fit(features, labels)

#preds = clf.predict(x_test)
#print("Acurracy: ", accuracy_score(y_test, preds))
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
#print(df)
#print(df.ix[:,0])
df.set_index('PassengerId', inplace=True)

df.to_csv('out.csv', sep='\t')

#print(result)
#print(len(result))

#print(df)

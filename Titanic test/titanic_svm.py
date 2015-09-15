import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

df = pd.read_csv(('train.csv'), header=0)

# convert male/female to 1/0
df['Gender'] = df['Sex'].map( {'female':0, 'male':1}).astype(int)

# convert embark S/C/Q to 0/1/2
df.loc[ df.Embarked.isnull(), 'Embarked'] = 'S'
df['EmbarkNum'] = df['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)

# fill in the missing age
median_ages = np.zeros((2, 3))

for i in range(0, 2):
  for j in range(0, 3):
    median_ages[i, j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):
  for j in range(0, 3):
    df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]
	

# drop the following columns
df = df.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
print df.dtypes

train_data = df.values
print train_data

#create the random forest object which will include all the parameters for this fit
forest = RandomForestClassifier(n_estimators = 1000)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])


# apply the same process to the test dataset

tf = pd.read_csv(('test.csv'), header=0)

# convert male/female to 1/0
tf['Gender'] = tf['Sex'].map( {'female':0, 'male':1}).astype(int)

# convert embark S/C/Q to 0/1/2
tf.loc[ tf.Embarked.isnull(), 'Embarked'] = 'S'
tf['EmbarkNum'] = tf['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)

# fill in the missing age

tf['AgeFill'] = tf['Age']

for i in range(0, 2):
  for j in range(0, 3):
    tf.loc[ (tf.Age.isnull()) & (tf.Gender == i) & (tf.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

# there is one person without price, come on!!!!
tf.loc[ (tf.Fare.isnull()), 'Fare'] = 0
	
# drop the following columns
tf = tf.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
print tf.dtypes

test_data = tf.values

print test_data

# take the same decision trees and run it on the test
output = forest.predict(test_data)
print output.shape

workDir = r'/Users/zhu/Documents/machine learning -kaggle/Titanic/'

prediction_file = open(workDir +'rfmodel.csv', 'wb')
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in range(len(output)):
  prediction_file_object.writerow([row + 892, output[row]])


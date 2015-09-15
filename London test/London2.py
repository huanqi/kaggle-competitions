# load the training dataset
import numpy as np
import csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation as cv



train_data = []
with open('./train.csv', 'r')as trainfile:
  trainReader = csv.reader(trainfile, delimiter =',')
  for row in trainReader:
    train_data.append(row)
train_data_array = np.array(train_data, dtype = float)
print train_data_array.shape

train_result = []
file_label = open('./trainLabels.csv', 'r')
data_label = csv.reader(file_label, delimiter = ",")
for row in data_label:
  train_result.append(row)
train_result_array = np.array(train_result, dtype = float)
print train_result_array.shape
file_label.close()

test_data = []
file_label = open('./test.csv', 'r')
data_test = csv.reader(file_label, delimiter = ",")
for row in data_test:
  test_data.append(row)
test_data_array = np.array(test_data, dtype = float)
print test_data_array.shape
file_label.close()

# normalize the features in the train and test dataset
train_data_array_norm = preprocessing.scale(train_data_array)
test_data_array_norm = preprocessing.scale(test_data_array)

# run the module of PCA
#pca = PCA(n_components = 10)
#train_data_array_norm_pca = pca.fit_transform(train_data_array_norm, train_result_array)
#test_data_array_norm_pca = pca.transform(test_data_array_norm)
#print 'train data shape', train_data_array_norm_pca.shape

# tree-based feature selection
classifier = ExtraTreesClassifier()
train_data_array_norm_pca = classifier.fit_transform(train_data_array_norm, np.ravel(train_result_array))
test_data_array_norm_pca = classifier.transform(test_data_array_norm)
print 'train data shape', train_data_array_norm_pca.shape


## build SVM
# random shuffle
np.random.seed(0)
indices = np.random.permutation(len(train_result_array))

classifer = svm.SVC(C=20, gamma = 0.05)

# cross validation
scores = cv.cross_val_score(classifier, train_data_array_norm_pca, np.ravel(train_result_array), cv = 30)


classifer.fit(train_data_array_norm_pca[indices[:-200], :], np.ravel(train_result_array[indices[:-200]]))


# calculate the test error and select the best model
test_prediction = classifer.predict(train_data_array_norm_pca[indices[-200:], :])
test_result = np.ravel(train_result_array[indices[-200:]])
count = 0
for i in range(200):
#  print test_prediction[i], test_result[i]
  if test_prediction[i] == test_result[i]:
    count = count + 1
print count
accuracy_rate = count/200.
print accuracy_rate

# predict the results on the test dataset
result = []
result = classifer.predict(test_data_array_norm_pca)
print result.shape

# save the predicted results
writer = csv.writer(open('./prediction.csv', 'wb'))
for row in result:
    writer.writerow([row])

  
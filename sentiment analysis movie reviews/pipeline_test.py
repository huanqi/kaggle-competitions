from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

def scikit_learn(train_set, train_labels):
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsOneClassifier(LinearSVC())),
                     ])
    X_train = np.asarray(train_set)
    X_lable = np.asarray(train_labels)
    text_clf = text_clf.fit(X_train, X_lable)
    return text_clf
	
	
train = pd.read_csv("train.tsv", header =0,\
                    delimiter = "\t", quoting=4)
					
text_clf = scikit_learn(train['Phrase'], train['Sentiment'])

test = pd.read_csv("test.tsv", header =0,\
                    delimiter = "\t", quoting=3)
					
prediction = text_clf.predict(test['Phrase'])

output = pd.DataFrame( data = {"PhraseId":test["PhraseId"], "Sentiment":prediction} )
output.to_csv( "sentiment_model.csv", index = False, quoting = 2)

print "program finised"




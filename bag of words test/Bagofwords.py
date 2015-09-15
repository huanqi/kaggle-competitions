import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def review_to_words (raw_review):
  # function to convert a raw review to a string of words
  review_text = BeautifulSoup(raw_review).get_text()
	# get rid of HTML tags
  letters_only = re.sub("[^a-zA-Z]", " ", review_text)
	# clean up the text, get rid of non character text
  lower_case = letters_only.lower()
	# convert to lower case
  words = lower_case.split()
	# convert to a list of individual words
  stops = set(stopwords.words('english'))
  meaningful_words = [w for w in words if not w in stops]
    # remove the stopwords
  return( " ".join(meaningful_words))
  

train = pd.read_csv("labeledTrainData.tsv", header =0,\
                    delimiter = "\t", quoting=3)
					
#clean_review = review_to_words(train['review'][0])
#print clean_review

num_reviews = train['review'].size
clean_train_reviews = []

print "Cleaning and parsing the training set movie reviews...\n"

for i in xrange(0, num_reviews):
  if( (i+1)%1000 ==0) :
    print "Review %d of %d \n" % (i+1, num_reviews)
  clean_train_reviews.append( review_to_words( train['review'][i]) )

vectorizer = CountVectorizer(analyzer = 'word', \
							tokenizer = None, \
							preprocessor = None, \
							stop_words = None, \
							max_features = 5000)
							
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print train_data_features.shape

print "Training the random forest..."
	# train random forest classifier with the data
forest = RandomForestClassifier(n_estimators = 500)
forest = forest.fit(train_data_features, train['sentiment'])

test = pd.read_csv('testData.tsv', header = 0, delimiter = '\t',\
					quoting = 3)
print test.shape
	# load the test data

num_reviews = len(test['review'])
clean_test_review = []

print "cleaning and parsing the test set movie reviews..."
for i in xrange(0,num_reviews):
  if ((i+1)%1000 == 0):
    print "Review %d of %d\n" % (i+1, num_reviews)
  clean_reviews = review_to_words( test['review'][i])
  clean_test_review.append(clean_reviews)
  
test_data_feature = vectorizer.transform(clean_test_review)
test_data_feature = test_data_feature.toarray()

result = forest.predict(test_data_feature)

output = pd.DataFrame( data = {"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_words_model.csv", index = False, quoting = 3)

					

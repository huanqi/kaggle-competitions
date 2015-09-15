import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk

train = pd.read_csv("labeledTrainData.tsv", header =0, delimiter = "\t", \
					quoting=3)

print train.shape
#print train.describe()
print train.head(3)	
print train.dtypes

#print train["review"][0]

# get rid of HTML tags
example1 = BeautifulSoup(train["review"][0])
#print train["review"][0]
#print example1.get_text()	

# clean up text with regular expression
letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text() )
#print letters_only

lower_case = letters_only.lower()
words = lower_case.split()
#print words[1]

from nltk.corpus import stopwords
#print stopwords.words('english')

words = [w for w in words if not w in stopwords.words('english')]
print words		
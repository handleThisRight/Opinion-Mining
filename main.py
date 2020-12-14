import pandas as pd 
import numpy as np

import csv

#Vectorizers
from sklearn.feature_extraction.text import *
# from sklearn.pipeline import FeatureUnion
# from zeugma.embeddings import EmbeddingTransformer


from sklearn.model_selection import train_test_split

#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC




################################################################################
# data loading and shrinking
################################################################################

def loadData(path, colX, colY):
	data = np.array(pd.read_csv(path))

	X = data[:,colX]
	y = data[:,colY]

	y[y == 'rotten'] = 0
	y[y == 'fresh'] = 1
	y = y.astype(float)
	X = X.astype(str)
	# print(X.shape,y.shape)
	# u, count = np.unique(y, return_counts=True)
	# print(count)
	return X,y


# shrinks the dataset keeping the class distribution same
# initial distribution 50:50 i.e. 240000 rotten + 240000 fresh
def shrink(X, y, size, random=True):
	size //= 2
	ifresh = np.nonzero(y==0)
	irotten = np.nonzero(y==1)

	if(random==False):
		np.random.seed(0)
		np.random.shuffle(ifresh)
		np.random.shuffle(irotten)

	freshX = X[ifresh][:size]
	freshy = y[ifresh][:size]

	rottenX = X[irotten][:size]
	rotteny = y[irotten][:size]

	X_shrink = np.concatenate((freshX,rottenX))
	y_shrink = np.concatenate((freshy,rotteny))
	return X_shrink, y_shrink



Xa, ya = loadData("./Dataset/rotten_tomatoes_reviews.csv", 1, 0)

Xb, yb = loadData("./preprocessed/withoutPunc.csv", 0, 1)

Xc, yc = loadData("./preprocessed/withPunc.csv", 0, 1)



Xa_red, ya_red = shrink(Xa, ya, 480000)
Xb_red, yb_red = shrink(Xb, yb, 480000)
Xc_red, yc_red = shrink(Xc, yc, 480000)

def vectorise(X, vectoriser):
	vectorised = vectoriser.fit_transform(X)
	print(vectorised.shape)
	return vectorised


def runmodel(X, y, model):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.30,random_state=1)
	model.fit(Xtrain, Ytrain)
	return model.score(Xtest,Ytest)



# Define Vectors
chargram = CountVectorizer(analyzer='char_wb', ngram_range=(3,6),max_df=0.5,min_df=0.01)
ngram = CountVectorizer(ngram_range=(1,4))
binary = CountVectorizer(binary=True)
tfid = TfidfVectorizer()
wordCount = CountVectorizer(binary=False)

vectors = [ngram]



#Define Models
MNB = MultinomialNB()
BNB = BernoulliNB()
rf = RandomForestClassifier(random_state=0)
logReg = LogisticRegression(random_state=0)
svm = SVC(random_state = 42)

models = [MNB, BNB, logReg]




def pipeline(X, y):
	for vector in vectors:
		X_vectorised = vectorise(X, vector)
		for model in models:
			print(runmodel(X_vectorised, y, model))




print("Using OG dataset")
pipeline(Xa_red, ya_red)
print()

print("preprocessed without punc")
pipeline(Xb_red, yb_red)
print()


# print("preprocessed with punc")
# pipeline(Xb_red, yb_red)
# print()
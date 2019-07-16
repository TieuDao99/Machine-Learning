import pandas as pd
import numpy as np 
import os 
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time 

df_idf = pd.read_json("data.json", lines=True)
def pre_process(headline):
	headline = headline.lower()                                    #lower_case
	headline = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", headline)      #remove_tags
	headline = re.sub("(\\d|\\W)+"," ", headline)                  #remove special characters and digits
	return headline
 
df_idf["headline"] = df_idf["headline"].apply(lambda x: pre_process(x))

def get_stop_words(stop_file_path):
	stop_set = set()
	with open(stop_file_path, 'r', encoding="utf-8") as fi:
		while True:
			stopword = fi.readline()
			if len(stopword) == 0: break
			stop_set.update([stopword.strip()])
	return frozenset(stop_set)
 
stopwords = get_stop_words("stop_words.data")       #load stopwords
documents = df_idf["headline"].tolist()             #get documents

cv = CountVectorizer(max_df=0.80, stop_words=stopwords) # create a vocabulary of words, ignore words that appear in
word_count_vector = cv.fit_transform(documents)           # 80% of documents eliminate stop words

#-------------------------------------------------------------------

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
feature_names = cv.get_feature_names()    

wd = np.full(len(feature_names), -1)
setY = np.array(df_idf["is_sarcastic"])

for i in range(len(documents)):
	tf_idf_vector = tfidf_transformer.transform(cv.transform([documents[i]]))
	temp = tf_idf_vector.tocoo().col
	label = setY[i]
	for j in temp:
		if wd[j] == 2: continue
		if wd[j] == -1: wd[j] = label; continue
		if wd[j] == 0 and label == 1: wd[j] = 2; continue
		if wd[j] == 1 and label == 0: wd[j] = 2; continue
"""
cnt0, cnt1, cnt2 = 0, 0, 0
for i in wd:
	if i == 0: cnt0 += 1
	if i == 1: cnt1 += 1
	if i == 2: cnt2 += 1
print(cnt0, cnt1, cnt2)
"""

setX = []
for i in range(len(documents)):
	tf_idf_vector = tfidf_transformer.transform(cv.transform([documents[i]]))
	temp = tf_idf_vector.tocoo()
	tempX = [0.0] * 3
	for idx, score in zip(temp.col, temp.data):
		tempX[wd[idx]] += score
	setX.append(tempX)

setX = np.array(setX)


trainingX = np.array(setX[0:20000])
testX = np.array(setX[20000:26709])
trainingY = np.array(setY[0:20000])
testY = np.array(setY[20000:26709])


start_time = time.time()


logreg = LogisticRegression()
logreg.fit(trainingX, trainingY)
predictY = logreg.predict(testX)



## -------------- cac mo hinh khac -------------
"""
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(trainingX, trainingY)
predictY = neigh.predict(testX)

"""
"""
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(trainingX, trainingY)
predictY = clf.predict(testX)
"""

# ---------------------------------------------- 

print(classification_report(testY, predictY))
print("Accuracy:", metrics.accuracy_score(testY, predictY))



end_time = time.time()
print('total run-time: %f ms' % ((end_time - start_time) * 1000))


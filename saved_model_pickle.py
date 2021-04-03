import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
import pandas as pd
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

url = "https://github.com/laiwenghong/CZ4034/blob/d87c7bce5b0b22f086801cfbcb0f29378b3b90b1/train.csv"
df = pd.read_csv(url)

#def remove_punctuation(text):
#    final = "".join(u for u in text if u not in ("?","'", ".", ";", ":",  "!",'"'))
#    return final

#preprocessing
#remove non-alphabetic characters, convert to lowercase, remove stopwords, stemming
#df['tweettextcleaned'] = df['tweettextcleaned'].apply(remove_punctuation)
df['tweettextcleaned'] = df['tweettextcleaned'].str.lower()
df['tweettextcleaned'] = df['tweettextcleaned'].str.split()
stop_words = set(stopwords.words('english')) 
df['tweettextcleaned'] = df['tweettextcleaned'].apply(lambda x: " ".join(x for x in x if x not in stop_words))
df['tweettextcleaned'] = df['tweettextcleaned'].apply(lambda x: word_tokenize(x))
stemmer = PorterStemmer()
df['tweettextcleaned'] = df['tweettextcleaned'].apply(lambda x: [stemmer.stem(y) for y in x])
df['tweettextcleaned'] = df['tweettextcleaned'].apply(lambda x: " ".join(x for x in x))

#count vectorizer
cv = CountVectorizer(analyzer=lambda x: x) #disable analyzer
df['cv'] = cv.fit_transform(df['tweettextcleaned'])

#tfidf conversion
tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
df['tfidf']=tfidf_vectorizer.fit_transform(df['tweettextcleaned'])

model = MultinomialNB()
model_name = model.__class__.__name__
tuned_parameters = {'class_prior': [[uniform.rvs(0,3), uniform.rvs(0,3)]]}
model = RandomizedSearchCV(MultinomialNB(), tuned_parameters, cv=5, 
scoring='f1_micro', n_iter=10)
grid_search.fit(X_train,y_train)
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)
print("Best Trainset Accuracy: %.2f%% using %s" % (grid_search.best_score_*100,grid_search.best_params_))
print("Testset Accuracy: %.2f%% " % (accuracy_score(y_test, y_test_pred)*100))
print("Classification Report of ",model_name,":\n",classification_report(y_test,y_test_pred))
print("Accuracy of",model_name,":",metrics.accuracy_score(y_test, y_test_pred))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

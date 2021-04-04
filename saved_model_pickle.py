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
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

url = "https://raw.githubusercontent.com/laiwenghong/CZ4034/main/train.csv"
col_names = ["tweettextcleaned","sentiment"]
df = pd.read_csv(url,  names = col_names)
df = df[df['sentiment']!=1]
df['tweettextcleaned'] = df['tweettextcleaned'].astype(str)

#preprocessing
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

tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['tweettextcleaned'])
X_train, X_test, y_train, y_test = train_test_split(text_tf, df['sentiment'], test_size=0.3, random_state=123)

model = RandomForestClassifier(n_estimators= 100, min_samples_split= 10, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 50, criterion= 'entropy', bootstrap= False)
model_name = model.__class__.__name__
grid_search = model.fit(X_train,y_train)
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)
print("Testset Accuracy: %.2f%% " % (accuracy_score(y_test, y_test_pred)*100))
print("Classification Report of ",model_name,":\n",classification_report(y_test,y_test_pred))
# print("Accuracy of",model_name,":",metrics.accuracy_score(y_test, y_test_pred))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
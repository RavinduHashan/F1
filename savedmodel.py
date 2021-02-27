import numpy as np
import pandas as pd

dataset = pd.read_csv('amazonreviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
corpus = []
for i in dataset['review']:
  review = re.sub('[^a-zA-Z]', ' ', i )
  review = review.lower()
  review = contractions.fix(review)
  review = word_tokenize(review)
  lemmatizer = WordNetLemmatizer()
  ps = PorterStemmer() 
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = [lemmatizer.lemmatize(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma= 'scale' , C=1 , random_state = 1)
classifier.fit(X_train, y_train)

import joblib
filename = 'finalized_model.sav'
joblib.dump(classifier, open(filename, 'wb'))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(accuracy_score)
import string
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

df = pd.read_csv(r"C:\Users\Pranat\Documents\Spam classifier\spam2_1.csv")
df['v2'] = df['v2'].astype('string')
df['v1'] = df['v1'].astype('string')
print(df.dtypes)

STOPWORDS = set(stopwords.words('english'))
def clean(data):
  data = data.lower()
  data = re.sub(r'[^0-9a-zA-Z]', ' ', data) ##match for letter or numbers and returns what is not i.e. weird characters.
  data = " ".join(word for word in data.split() if word not in STOPWORDS)
  return data 
df['clean_messages'] = df['v2'].apply(clean)
x = df['clean_messages'].astype("string")
print(x)
y = df['v1']

def classify(model,x,y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=17, shuffle=True, stratify=y) ##stratifies the data with class labels as y.
  pipeline_model = Pipeline([('vect', CountVectorizer()),('tfidf',TfidfTransformer()),('clf', model)])
  pipeline_model.fit(x_train, y_train)
  
  print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
  y_pred = pipeline_model.predict(x_test)
  print(classification_report(y_test, y_pred))

#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors=2)
#classify(model,x,y) ####92% accuracy

#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
#classify(model, x, y)#### 97% accuracy

from sklearn.svm import SVC
model = SVC() ###using rbf(radial basis function) kernel, extremly similar to KNN algo it is based on cosine similarity between points and distance between two points like KNN.
classify(model, x, y) ###98% accuracy.
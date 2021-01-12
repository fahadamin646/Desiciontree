import numpy as np # linear algebra
import pandas as pd
df = pd.read_json('Dataset for Detection of Cyber-Trolls.json', lines= True)
df.head()
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,classification_report,accuracy_score
corpus = ['I am Happy']

for i in range (0, len(df)):                                #Iterating over each review
    review = re.sub('[^a-zA-Z]',' ',df['content'][i])       #Removing annotations
    review = review.lower()                                 #Converting everything to lower case
    review = review.split()                                 #Splitting each word in a review into a separate list
    review = ' '.join(review)
    r=review                               #Joining all the words into a single list
    corpus.append(review)
    
bow_transformer =  CountVectorizer()               #Creating our vectorizer
bow_transformer = bow_transformer.fit(corpus)      #Fitting the vectorizer to our reviews
messages_bow = bow_transformer.transform(corpus)  #Transforming to a sparse format (To use it for our training)
tfidf_transformer = TfidfTransformer().fit(messages_bow)                #Applying TF-ID to our reviews
X = tfidf_transformer.transform(messages_bow)                        #Transforming to a sparse format(For training purposes)
#print(X)
y = []
for i in range(0,len(df)):
    y.append(df.annotation[i]['label'])                               #Extracting labels from our dataset (From the dictionary)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   #Splitting train and test data


dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(X_train,y_train)
# print(dct.predict(X[0]))
#print(X_test)

with open('model','wb') as f:
  pickle.dump(dct,f)

# with open('model', 'rb') as f:
#     ppn = pickle.load(f)
# print(ppn.predict(X[0]))
y_pred  =dct.predict(X_test)
# print(y_pred)
print(classification_report(y_test,y_pred)) 
print(accuracy_score(y_test, y_pred))
# %%
#Import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from wordcloud import WordCloud
from random import randint

#Uncomment if not available and download it.
#nltk.download('stopwords')
#nltk.download('punkt')

#Load the dataset
df = pd.read_csv("../Data/Combined_News_DJIA.csv", encoding = 'ISO-8859-1')

#Print first 5 rows
df.head()

# %%
#Cleaning

#Drop NaN 
df.dropna(inplace=True)

#Visualize label distribution
plt.figure(figsize=(7, 4))
sns.countplot(x='Label', data=df)
plt.xlabel('Stock Sentiments (0-Down(Negative), 1-Up (Positive))')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

# %%
#Clean and Preprocessing

#We will merge all the  top 25 headlines in one string per row
headlines = []
for row in range(0, len(df.index)):
    headlines.append(' '.join(str(x) for x in df.iloc[row, 2:]))

#Remove stop words and save it in corpus
ps = PorterStemmer()
corpus = []
for text in headlines:
    words = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(words))

# %%
#Train and Test split using date range
train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']

train_corpus = corpus[:len(train)]
test_corpus = corpus[len(train):]

y_train = train['Label']
y_test = test['Label']

# Feature extraction
cv = CountVectorizer(max_features=10000, ngram_range=(2, 2))
X_train = cv.fit_transform(train_corpus).toarray()
X_test = cv.transform(test_corpus).toarray()

# %%
# Model training and evaluation
from sklearn.metrics import f1_score
#Define a common function to evaluate all the models
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {round(acc * 100, 2)} %")
    print(f"Precision: {round(prec, 2)}")
    print(f"Recall: {round(rec, 2)}")
    print(f"F1 Score: {round(f1, 2)}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Run models
evaluate_model(LogisticRegression(), "Logistic Regression")
evaluate_model(RandomForestClassifier(n_estimators=100, criterion='entropy'), "Random Forest")
evaluate_model(GaussianNB(), "Naive Bayes")
evaluate_model(MLPClassifier(hidden_layer_sizes=(100,), max_iter=300), "MLP Classifier")

# %%
# Predict a random test sample
sample_test = df[df['Date'] > '2014-12-31'].reset_index()
sample_news = sample_test.iloc[randint(0, sample_test.shape[0] - 1), 2:]
sample_text = ' '.join(str(x) for x in sample_news)
sample_text = re.sub('[^a-zA-Z]', ' ', sample_text).lower()
words = [ps.stem(w) for w in sample_text.split() if w not in set(stopwords.words('english'))]
final_text = ' '.join(words)

print(final_text)

# %%
#Predict
temp_vec = cv.transform([final_text]).toarray()
pred = LogisticRegression().fit(X_train, y_train).predict(temp_vec)[0]
print("Prediction:", "Stock Up" if pred else "Stock Down")

# %%




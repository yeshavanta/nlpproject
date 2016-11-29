__author__ = 'ykp'

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import f1_score

data = []
i = 0
reviews = []
labels = []
features = []

with open('data/yelp_academic_dataset_review.json') as f:
    for line in f:
        if i >= 100000:
            break
        a = json.loads(line)
        reviews.append((a['text'],a['stars']))
        i += 1

corpus = []
labels = []

for i,j in reviews:

    corpus.append(i)

    if j == 5 or j == 4:
        labels.append('p')
    elif j == 3:
        labels.append('n')
    else:
        labels.append('ne')

vectorizer = CountVectorizer(min_df=1,stop_words='english',lowercase=True,max_features=100,ngram_range=(3,1))
X = vectorizer.fit_transform(corpus).toarray()
train_data = X[:75000]
train_labels = labels[:75000]
test_data = X[75000:]
test_labels = labels[75000:]

model = svm.SVC(kernel='linear',C = 1,gamma = 1)
model.fit(train_data, train_labels)
y_pred = model.predict(test_data)
accuracy = model.score(test_data,test_labels)

print(accuracy * 100)
print(f1_score(test_labels, y_pred, average='weighted') * 100)
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier


data = []
i = 0
reviews = []
labels = []
features = []

with open('data/yelp_academic_dataset_review.json') as f:
    for line in f:
        if i >= 10000:
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
train_data = X[:7500]
train_labels = labels[:7500]
test_data = X[7500:]
test_labels = labels[7500:]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


clf.fit(train_data, train_labels)
y_pred = clf.predict(test_data)
accuracy = clf.score(test_data,test_labels)

print(accuracy * 100)
print(f1_score(test_labels, y_pred, average='weighted') * 100)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import string
import util


reviews = []
labels = []
text = []

reviews = util.readFile('data/yelp_academic_dataset_review.json')

#
# Here we are going through the reviews that we read from the yelp academic data set,
# and we are replacing the star ratings with labels
#
#   1 ,2 = "ne"
#      3 = "n"
#   4, 5 = "p"
#
for i,j in reviews:

    text.append(i)

    if j == 5 or j == 4:
        labels.append('p')
    elif j == 3:
        labels.append('n')
    else:
        labels.append('ne')

#
#   Feature Extraction along with:
#   a. Stop word removal
#   b. bigram and trigram generation
#   c. obtaining top 100 features based on document frequency
#
#

vectorizer = CountVectorizer(min_df=1,stop_words='english',lowercase=True,max_features=100,ngram_range=(3,1))
X = vectorizer.fit_transform(text).toarray()


#
#
#   Creating training data, training labels and testing data, testing labels
#
#

trainingData = X[:75000]
trainingLabels = labels[:75000]
testingData = X[75000:]
testingLabels = labels[75000:]

#
#   Getting the instance of a MLP perceptron
#
clf = MLPClassifier(solver='adam', alpha=1e-10, hidden_layer_sizes=(56, 27), random_state=4)

#
# training the model using training data and training labels
#
clf.fit(trainingData, trainingLabels)

#
#   predicting the values for testingData
#
pred = clf.predict(testingData)

#
#   calculating the F1 score using the predicted labels
#
print(f1_score(testingLabels, pred, average='weighted') * 100)






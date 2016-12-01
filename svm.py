__author__ = 'ykp'

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import f1_score

data = []
reviews = []
corpus = []
labels = []
X = []
model = {}

def readDataFromYelpDataBase():
    global reviews
    i = 0
    file = open('data/yelp_academic_dataset_review.json')
    for line in file:
        if i >= 100000:
            break
        a = json.loads(line)
        reviews.append((a['text'],a['stars']))
        i += 1


def generateCorpusAndLabels():
    global corpus
    global labels
    for i,j in reviews:
        corpus.append(i)
        if j == 5 or j == 4:
            labels.append('p')
        elif j == 3:
            labels.append('n')
        else:
            labels.append('ne')

def extractFeatures():
    global X
    vectorizer = CountVectorizer(min_df=1,stop_words='english',lowercase=True,max_features=100,ngram_range=(3,1))
    X = vectorizer.fit_transform(corpus).toarray()

def populateTrainAndTestData():
    global X
    global trainingData
    global trainingLabels
    global testingData
    global testingLabels
    trainingData = X[:75000]
    trainingLabels = labels[:75000]
    testingData = X[75000:]
    testingLabels = labels[75000:]

def trainTheModel():
    global trainingData
    global testingData
    global model
    model = svm.SVC(kernel='linear',C = 1,gamma = 10)
    model.fit(trainingData, trainingLabels)

def predictAndScore():
    global  model
    y_pred = model.predict(testingData)
    accuracy = model.score(testingData,testingLabels)
    print(accuracy * 100)
    print(f1_score(testingLabels, y_pred, average='weighted') * 100)


if __name__ == '__main__':
    readDataFromYelpDataBase()
    generateCorpusAndLabels()
    extractFeatures()
    populateTrainAndTestData()
    trainTheModel()
    predictAndScore()





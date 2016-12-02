import json
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayesClassifier(object):
    def __init__(self):
        self.reviews = []
        self.labels = []
        self.features = []
        self.corpus = []
        self.labels = []
        self.data = []
        self.trainingModel = []
        self.train_list = []
        self.train_labels = []
        self.test_list = []
        self.train_list = []

    def readDataSet(self):
        with open('data/yelp_academic_dataset_review.json') as file:
            for line in file:
                if i >= 100000:
                    break
                arr = json.loads(line)
                self.reviews.append((arr['text'],arr['stars']))
                i += 1
                
    def getFeatures(self):
        vectorizer = CountVectorizer(min_df=1,stop_words='english',lowercase=True,max_features=100,ngram_range=(2,1))
        self.data = vectorizer.fit_transform(corpus).toarray()
        
    def assignLabels(self):
        for i,j in self.reviews:
            self.corpus.append(i)
            if j == 5 or j == 4:
                self.labels.append('p')
            elif j == 3:
                self.labels.append('n')
            else:
                self.labels.append('ne')

    def splitCorpusAsTrainAndTest(self):
        self.train_list = self.data[:75000]
        self.train_labels = self.labels[:75000]
        self.test_list = self.features[75000:]
        self.test_labels = self.labels[75000:]

    def trainNaiveBayes(self):
        self.trainingModel = GaussianNB()
        self.trainingModel.fit(train_data,train_labels)
        
    def predict(self):
        y_pred = self.trainingModel.predict(test)
        accuracy = clf1.score(test,test_labels)
        f1_score(test_labels, y_pred, average='weighted')

def main():
    nb = NaiveBayesClassifier()
    nb.readDataSet()
    nb.assignLabels()
    nb.getFeatures()
    nb.splitCorpusAsTrainAndTest()
    nb.trainNaiveBayes()
    nb.predict()
    
if __name__ == '__main__':
    main()

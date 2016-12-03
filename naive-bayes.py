import json
import string
import nltk
nltk.download('punkt')
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

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
        
    def stemTokens(self,tokens,stemmer):
        stemList = []
        for w in tokens:
            stemList.append(stemmer.stem(w))
        return stemList
    
    def tokenize(self,text):
        tokens = nltk.word_tokenize(text)
        tokens = [i for i in tokens if i not in string.punctuation]
        stemmedTokens = self.stemTokens(tokens, stemmer)
        return stemmedTokens

    def readDataSet(self):
        with open('data/yelp_academic_dataset_review.json') as file:
            i = 0
            for line in file:
                if i >= 100000:
                    break
                arr = json.loads(line)
                self.reviews.append((arr['text'],arr['stars']))
                i += 1
                
    def getFeatures(self):
        vectorizer = CountVectorizer(analyzer='word',strip_accents='ascii',tokenizer=self.tokenize,min_df=30,stop_words='english',lowercase=True,max_features=1000,ngram_range=(1,3))
        self.data = vectorizer.fit_transform(self.corpus).toarray()
        
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
        self.test_list = self.data[75000:]
        self.test_labels = self.labels[75000:]

    def trainNaiveBayes(self):
        self.trainingModel = GaussianNB()
        self.trainingModel.fit(self.train_list,self.train_labels)
        
    def predict(self):
        y_pred = self.trainingModel.predict(self.test_list)
        accuracy = self.trainingModel.score(self.test_list,self.test_labels)
        print(accuracy)
        print(f1_score(self.test_labels, y_pred, average='weighted'))

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

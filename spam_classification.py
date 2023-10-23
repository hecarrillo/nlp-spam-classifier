import numpy as np
import unicodedata
import nltk
import matplotlib.pyplot as plt

# Text Preprocessing class
class TextPreprocessing:
    def __init__(self):
        pass

    def getMessageTag(self, line): 
        line = line.replace('\n', '')
        if line.endswith(",ham"):
            return 1
        elif line.endswith(",spam"):
            return 0
        else: 
            print("Error: Invalid line format")
            # print line with invisible characters
            print(line.encode('utf-8'))

    def cleanMessage(self, line): 
        line = line.replace('\n', '')
        line = line.replace(',ham', '')
        line = line.replace(',spam', '')
        line = self.remove_accented_chars(line)
        line = line.lower()
        return line

    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def tokenize_and_lemmatize_file(self, file):
        with open(file, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
            clean_lines = [self.cleanMessage(line) for line in lines]
            tokenized_lines = [nltk.word_tokenize(line) for line in clean_lines]
            return [" ".join([self.lemmatize_word(word) for word in line]) for line in tokenized_lines]

    def lemmatize_word(self, word):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)

    def get_vocabulary_from_tokenized_lines(self, tokenized_lines):
        vocabulary = set()
        for line in tokenized_lines:
            for word in line.split():
                vocabulary.add(word)
        return vocabulary

# TDM class
class TDM:
    def __init__(self, vocabulary):
        self.vocabulary = list(vocabulary)

    def transform(self, lines):
        tdm = np.zeros((len(lines), len(self.vocabulary)))
        for i, line in enumerate(lines):
            words = line.split()
            for word in words:
                if word in self.vocabulary:
                    tdm[i][self.vocabulary.index(word)] += 1
        return tdm

# Custom Logistic Regression
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.3, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)
            
            cost = (-1/n_samples) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            self.costs.append(cost)

            # print(f"Iteration {_+1}/{self.n_iterations}, Cost: {cost}")

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)
        return [1 if i > 0.5 else 0 for i in predictions]

# Main Execution
INPUT_TRAINING_FILE_NAME = './inputs/training.txt'
INPUT_TESTING_FILE_NAME = './inputs/testing.txt'

text_preprocessing = TextPreprocessing()
training_lines = text_preprocessing.tokenize_and_lemmatize_file(INPUT_TRAINING_FILE_NAME)
testing_lines = text_preprocessing.tokenize_and_lemmatize_file(INPUT_TESTING_FILE_NAME)

training_vocabulary = text_preprocessing.get_vocabulary_from_tokenized_lines(training_lines)
testing_vocabulary = text_preprocessing.get_vocabulary_from_tokenized_lines(testing_lines)

tdm_transformer = TDM(training_vocabulary)
training_tdm = tdm_transformer.transform(training_lines)
testing_tdm = tdm_transformer.transform(testing_lines)

training_labels = [text_preprocessing.getMessageTag(line) for line in open(INPUT_TRAINING_FILE_NAME, 'r', encoding='utf-8').readlines()]
testing_labels = [text_preprocessing.getMessageTag(line) for line in open(INPUT_TESTING_FILE_NAME, 'r', encoding='utf-8').readlines()]

lr = LogisticRegressionCustom()
lr.fit(training_tdm, training_labels)
predicted_labels = lr.predict(testing_tdm)

accuracy = np.mean(np.array(predicted_labels) == np.array(testing_labels))

# print confusion matrix
print('Confusion Matrix:')
print('-----------------')
print('True Positive: ', np.sum(np.array(predicted_labels) & np.array(testing_labels)))
print('False Positive: ', np.sum(np.array(predicted_labels) & np.logical_not(np.array(testing_labels))))
print('False Negative: ', np.sum(np.logical_not(np.array(predicted_labels)) & np.array(testing_labels)))
print('True Negative: ', np.sum(np.logical_not(np.array(predicted_labels)) & np.logical_not(np.array(testing_labels))))
print('-----------------')

print(f"Accuracy: {accuracy:.4f}")



# Plotting cost over iterations
plt.plot(range(lr.n_iterations), lr.costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over Iterations")
plt.show()

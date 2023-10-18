import unicodedata
import nltk
import random
import re
from contractions import contractions_dict
import spacy
# nlp = spacy.load('en_core_web_sm')
class text_preprocessing:
    def __init__(self):
        pass

    def getMessageTag(self, line): 
        if line.endswith(",ham"):
            return 1
        elif line.endswith(",spam"):
            return 0

    def cleanMessage(self, line): 
        line = line.replace('\n', '')
        line = line.replace(',ham', '')
        line = line.replace(',spam', '')
        line = self.remove_accented_chars(line)
        line = line.lower()
        return line

    def simple_stemmer(self, word):
        ps = nltk.porter.PorterStemmer()
        return ps.stem(word)

    def lemmatize_word(self, word):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)


    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def tokenize_and_lematize_file(self, file):
        with open(file, 'r', encoding='windows-1252') as input_file:
            lines = input_file.readlines()
            clean_lines = [self.cleanMessage(line) for line in lines]
            tokenized_lines = [nltk.word_tokenize(line) for line in clean_lines]
            return [[self.lemmatize_word(word) for word in line] for line in tokenized_lines]
    
    def get_vocabulary_from_tokenized_lines(self, tokenized_lines):
        vocabulary = set()
        for line in tokenized_lines:
            for word in line:
                vocabulary.add(word)
        return vocabulary
    
# main 
INPUT_TRAINING_FILE_NAME = './inputs/training.txt'
INPUT_TESTING_FILE_NAME = './inputs/testing.txt'

text_preprocessing = text_preprocessing()
training_lines = text_preprocessing.tokenize_and_lematize_file(INPUT_TRAINING_FILE_NAME)
testing_lines = text_preprocessing.tokenize_and_lematize_file(INPUT_TESTING_FILE_NAME)

training_vocabulary = text_preprocessing.get_vocabulary_from_tokenized_lines(training_lines)
testing_vocabulary = text_preprocessing.get_vocabulary_from_tokenized_lines(testing_lines)

frequency_matrix_
        



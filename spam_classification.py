import unicodedata
import nltk
import random
import re
from contractions import contractions_dict
import spacy
# nlp = spacy.load('en_core_web_sm')

# import the input txt to tokenize from input/ folder
INPUT_FILE_NAME = 'SMS_Spam_Corpus_big.txt'
INPUT_FOLDER = 'inputs/'
OUTPUT_FOLDER = 'output/'
INPUT_FILE_PATH = INPUT_FOLDER + INPUT_FILE_NAME

# import the output txt to write the tokenized words to
OUTPUT_FILE_NAME = 'SMS_Spam_Corpus_big_tokenized.txt'
OUTPUT_FILE_PATH = OUTPUT_FOLDER + OUTPUT_FILE_NAME

def getMessageTag(line): 
    if line.endswith(",ham"):
        return 1
    elif line.endswith(",spam"):
        return 0

def cleanMessages(line): 
    line = line.replace('\n', '')
    line = line.replace(',ham', '')
    line = line.replace(',spam', '')
    line = remove_accented_chars(line)
    line = line.lower()
    return line

def simple_stemmer(word):
    ps = nltk.porter.PorterStemmer()
    return ps.stem(word)

def lemmatize_word(word):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return lemmatizer.lemmatize(word)


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# import the txt to read each line with windows 1252 encoding
with open(INPUT_FILE_PATH, 'r', encoding='windows-1252') as input_file:
    # read each line in the txt and store it in a list
    lines = input_file.readlines()
    # for each message, get the message tag and store it in a list
    messageTags = [getMessageTag(line.replace('\n', '')) for line in lines]
    # clean each line from unwanted new lines and tags
    cleanLines = [cleanMessages(line) for line in lines]
    # for each line, tokenize the sentences using nltk
    sentenceTokensPerMessage = [nltk.sent_tokenize(line) for line in cleanLines]
    # for each line, tokenize the words using nltk 
    wordTokensPerMessage = [nltk.word_tokenize(line) for line in cleanLines]
    stemmedWordTokensPerMessage = [
        [simple_stemmer(word) for word in message]
        for message in wordTokensPerMessage
    ]
    lemmatizedWordTokensPerMessage = [
        [lemmatize_word(word) for word in message]
        for message in wordTokensPerMessage
    ]
    print("number of lines: ", len(lines))
    randomInitialIndex = random.randrange(0, len(lines) - 10)
    randomFinalIndex = randomInitialIndex + 10

    print("\n Looking at indexes: ", randomInitialIndex, " to ", randomFinalIndex)
    print("\n Random 10 messages: \n", cleanLines[randomInitialIndex:randomFinalIndex])
    print("\n Tags of those messages: \n", messageTags[randomInitialIndex:randomFinalIndex])
    print("\n Sentences in the first 5 messages: \n", sentenceTokensPerMessage[randomInitialIndex:randomFinalIndex - 5])
    print("\n Words in the first 5 messages: \n", wordTokensPerMessage[randomInitialIndex:randomFinalIndex - 5])
    print("\n Stemmed words in the first 5 messages: \n", stemmedWordTokensPerMessage[randomInitialIndex:randomFinalIndex - 5])
    print("\n Lemmatized words in the first 5 messages: \n", lemmatizedWordTokensPerMessage[randomInitialIndex:randomFinalIndex - 5])
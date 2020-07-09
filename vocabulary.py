from nltk.corpus import stopwords
from os import listdir
from collections import Counter
import re
import string


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def process_docs(directory, vocab):
    lines = list()
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)


def save_list(lines, filename):
    lines = "\n".join(lines)
    file = open(filename, 'w')
    file.write(lines)
    file.close()


vocab = Counter()
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)

min_occurrence = 2

tokens = [k for k, c in vocab.items() if c >= min_occurrence]

save_list(tokens, 'vocab.txt')

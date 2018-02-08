import csv
import sys

import numpy as np
import pandas as pd

from Document import Document
from Stats import Stats

global stats

def _compute_indicator_score(term, clas):
    num = stats.count_term_in_clas(term, clas) + 1
    denom = stats.count_terms_in_clas(clas) + len(stats.vocab)
    return num / denom

def _compute_score(document, clas):
    score = np.log(stats.count_docs_with_clas(clas) / len(stats.documents))
    for term in document._terms:
        ind = np.log(_compute_indicator_score(term, clas))
        score += ind
    return score

def train(train_data):
    global stats
    stats = Stats(train_data)

def predict(test_data):
    with open('predictions.csv', 'w', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(['document', 'predict_class', 'predict_score', 'exp_predict_score'])
        for instance in test_data.iterrows():
            doctext = instance[1]['document']
            doc = Document(doctext)
            predict_clas = max(stats.classes, key=lambda c: _compute_score(doc, c))
            predict_score = _compute_score(doc, predict_clas)
            exp_predict_score = np.exp(predict_score)
            writer.writerow([doctext, predict_clas, predict_score, exp_predict_score])

def print_help():
    pass

def main():
    for arg in sys.argv:
        if arg in ('-h', '--help'):
            print_help()
    
    train_data = pd.read_csv('training_data.csv')
    test_data = pd.read_csv('test_data.csv')

    train(train_data)
    predict(test_data)

if __name__ == '__main__':
    main()

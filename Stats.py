from Document import Document

class Stats(object):
    def __init__(self, train_data):
        self.classes = set(train_data['class'])
        self.documents = list(map(lambda text: Document(text), train_data['document']))
        self.vocab = set([term for doc in self.documents for term in doc._terms])

        self._frame = train_data
    
    def count_terms_in_clas(self, clas):
        count = 0
        for row in self._frame.iterrows():
            if row[1]['class'] == clas:
                count += len(Document(row[1]['document'])._terms)
        return count
    
    def count_term_in_clas(self, term, clas):
        count = 0
        for row in self._frame.iterrows():
            if row[1]['class'] == clas:
                count += Document(row[1]['document'])._terms.count(term)
        return count
    
    def count_docs_with_clas(self, clas):
        count = 0
        for row in self._frame.iterrows():
            if row[1]['class'] == clas:
                count += 1
        return count

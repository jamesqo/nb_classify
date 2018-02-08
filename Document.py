class Document(object):
    def __init__(self, text):
        self._terms = text.split()

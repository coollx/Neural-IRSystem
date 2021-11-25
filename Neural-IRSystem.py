from Tokenizer import *
from Document import *

class IRSystem:
    def __init__(self, doc_path):
        #document path
        self.doc_path = doc_path
        
        #total number of documents in system
        self.N = 0
        
        #set of vocabulary
        self.vocabulary = set()
        
        #map of each document with it's ID
        self.doc_map = dict()
        
        #map of document frequency of each term
        self.doc_freq = dict()
        
        #as you can see
        self.inverted_index = dict()
        
        #average document length
        self.average_doc_length = 0
        
        self.index_documents()
        
    def index_documents(self):
         pass   

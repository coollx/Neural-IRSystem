import os
import gensim.models
from gensim import utils
from Document import *

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.docs = []
        with open(self.corpus_path, encoding='utf-8') as f:
            docs = f.readlines()
            for line in docs:
                #extract doc id and doc text
                id, text = line.split('\t', 1)
                #append document list
                self.docs.append(Document(id, text))

    def __iter__(self):
        for doc in self.docs:
            # assume there's one document per line, tokens separated by whitespace
            yield doc.raw_token_list


def train_Word2Vec(corpus_path = 'files' + os.sep + 'Trec_microblog11.txt',
                   saved_model_path = 'models' + os.sep + 'word2vec.txt'
                   ):
    
    #train word 2 vec model
    sentences = MyCorpus(corpus_path = corpus_path)
    model = gensim.models.Word2Vec(sentences=sentences, min_count = 1, vector_size = 200)
    model.save(saved_model_path)
    
    # #save the trained model to disk
    # with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    #     temporary_filepath = tmp.name
    #     model.save(temporary_filepath)
        
if __name__ == '__main__':
    #train_Word2Vec()
    model = gensim.models.Word2Vec.load('models' + os.sep + 'word2vec.txt')
    print(model.wv.most_similar(['BBC', 'World', 'Service'], topn=5))
    #print(model.wv['bbc'])
    print('bbc' in model.wv.index_to_key)
    q = 'hehe half-sister'
    print([t for t in q.split() if t in model.wv.index_to_key])
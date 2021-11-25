from Tokenizer import *
from math import *

class Document:
    tkn = Tokenizer()
    
    def __init__(self, id, text):
        #document ID
        self.id = id
        #document's raw text
        self.raw_text = text
        #populate document's token list
        self.token_list = Document.tkn.get_tokens(text)
        #tf-idf vector
        self.tf_idf = []
        #norm of tf-idf vector
        self.norm = 0
        #token frequency map
        self.freq_map = dict()
        
        for token in self.token_list:
            self.freq_map[token] = self.freq_map.get(token, 0) + 1
        
    # def get_id(self):
    #     return self.id
    
    # def get_text(self):
    #     return self.raw_text
    
    # def get_token_list(self):
    #     return self.token_list
    
    # def get_freq_map(self):
    #     return self.freq_map
    
    # def size(self):
    #     return len(self.token_list)
    
    def calc_norm(self, doc_freq, N):
        for token in self.token_list:
            self.tf_idf.append(
                (1 + log10(0.0 + self.freq_map[token])) * log10((0.0 + N) / (0.0 + doc_freq[token]))
            )
        self.norm = sqrt(sum(i*i for i in self.tf_idf))
    
    # def get_norm(self):
    #     return self.norm
    
    def __str__(self):
        return "Doc# {}: {}".format(self.get_id(), str(self.token_list))
    
    def __lt__(self, other):
        return other.norm > self.norm
    

if __name__ == '__main__':
    d = Document(1, 'hello Hello , ,,,\n  \
                 \
                 \
                 \
                 world "I am tired! I like fruit...and milk 大西瓜')
    d1 = Document(1, 'Yay its one of my favorite episodes of Boy Meets World :)')
    print(d.freq_map)
    print(d.tf_idf)
    
    from queue import PriorityQueue
    pq=PriorityQueue()
    pq.put((1,d))
    pq.put((2,d1))
    
    print(pq.get()[1].raw_text)
    
from Tokenizer import *
from Document import *
from math import *
from QueryParser import *
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import torch


class IRSystem:
    tkn = Tokenizer()
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device = 'cuda')
    
    def __init__(self, doc_path = 'files' + os.path.sep + 'Trec_microblog11.txt'):
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
        
        self.process_documents()
        
    def process_documents(self):
        with open(self.doc_path, encoding='utf-8') as f:
            docs = f.readlines()
            for line in docs:
                #extract doc id and doc text
                id, text = line.split('\t', 1)
                #create document object
                doc = Document(id, text)
                #extend vocabulary with current document
                self.vocabulary |= set(doc.token_list)
                #extend inverted index with current document
                self.index_document(doc)
                #add document into doc map
                self.doc_map[id] = doc
                #add document length to average
                self.average_doc_length += len(doc.token_list)
                
                self.N += 1
                
                #print(len(self.vocabulary))
            

        for doc in self.doc_map.values():
            doc.calc_norm(self.doc_freq, self.N)
            #print(doc.norm)
        
        self.average_doc_length /= self.N
        print("Indexing complete, total vocabulary = {} words".format(str(len(self.vocabulary))))
        
    def index_document(self, doc):
        for token in doc.freq_map.keys():
            #for every element in frequency map, increment it's document frequency by 1
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
            
            #update inverted index
            #insert a new array list to our hashmap if we don't have an associate list with that token.
            if not token in self.inverted_index:
                self.inverted_index[token] = list()
            
            #update the number of occurence of each term in inverted index
            self.inverted_index[token].append((doc, doc.freq_map[token]))
            
    def get_freq_map(self, text):
        #create a map to store each word in the document with it's frequency
        ret = dict()
        for token in text:
            ret[token] = ret.get(token, 0) + 1
        return ret
    
    def retrive_top_K(self, q, K, method = 'tf-idf'):
        #store each document and it's word vector
        similarity_map = dict()
        #remove stop words, tokenization using porter stemmer
        query = IRSystem.tkn.get_tokens(q)
        #get frequency map of the query
        query_freq_map = self.get_freq_map(query)
        
        #get max frequency from query
        query_max_freq = max(query_freq_map.values())
        
        #norm of the query
        query_norm = 0
        
        for term in query:
            if not term in self.inverted_index.keys():
                continue
            
            #compute idf of term by document frequency
            df_t = self.doc_freq[term]
            idf = log10(self.N / df_t)
            #compute w_t_q: weight of term in query
            if method == 'bm-25':
                w_t_q = log(1 + (0.5 + self.N - df_t) / (0.5 + df_t))
            #default method is tf-idf
            else:     
                w_t_q = 0.5 + 0.5 * query_freq_map[term] / query_max_freq * idf
                
            query_norm += w_t_q * w_t_q
            
            for doc, tf in self.inverted_index[term]:
                #calculate tf_idf of term t to document d
                if method == 'bm-25':
                    k = 0.3 #chosen via grid search
                    b = 0.5
                    w_t_d = ((1.0 + k) * tf)/(0.0 + tf + k*(1 - b + b * len(doc.token_list) / self.average_doc_length))
                #default method is tf-idf
                else:
                    #w_t_d = self.tf_idf(tf, idf)
                    w_t_d = (1 + log10(tf)) * idf
                    
                #update the consine score, update similarityMap
                similarity_map[doc] = similarity_map.get(doc, 0) + w_t_d * w_t_q
          
        #compute query norm
        query_norm = sqrt(query_norm)
               
        #use a list to store all retrived documents
        ret = list()
        
        #insert every element in the map to priority queue to get top k elements
        for doc in similarity_map.keys():
            if method == 'bm-25':
                ret.append((similarity_map[doc], doc))
            else:
                ret.append((similarity_map[doc] / doc.norm / query_norm, doc))
        
        ret.sort(key = lambda x : x[0], reverse = True)
        return ret[:K]
        
    
    def rerank(self, q, to_rerank):
        '''
        q: raw query
        to_rerank = [[score1, doc1],[score2, doc2], ...]
        '''
        if not torch.cuda.is_available():
            print("Warning: No GPU found. Please get a RTX3090")
        
        cross_inq = [[q, doc.raw_text] for (_, doc) in to_rerank]
        cross_scores = self.cross_encoder.predict(cross_inq)
        
        for i in range(len(to_rerank)):
            to_rerank[i] = (cross_scores[i], to_rerank[i][1])
        
        to_rerank.sort(key = lambda x : x[0], reverse = True)
    
    
    def run_query(self, 
                  query_file_path, output_file_path,
                  K = 1000, eval = False, method = 'bm-25', refine = True, rerank = True
                ):
        query_list = QueryParser().parse(query_file_path)
        with open(output_file_path, 'w', encoding = 'utf-8') as f:
            query_number = 1
            for query in query_list:
                #process extension query, retrive top 10 ranked query
                if refine:
                    refined_query = query
                    topRank = self.retrive_top_K(query, 10, method)
                    #append extended query to original query
                    for socore, doc in topRank:
                        refined_query += " " + doc.raw_text
                    
                res = self.retrive_top_K(refined_query, K, method)
                
                #it's cricial to use the original query to rerank, do not use the refinded query!
                if rerank:
                    self.rerank(query, res) 
                
                rank = 1
                for score, doc in res:
                    if eval:
                        f.write("{:d} Q0 {} {:d} {:.3f} muRun\n".format(query_number, doc.id, rank, score))
                    else:
                        f.write("MB{:03d} Q0 {:s} {:d} {:.3f} muRun\n".format(query_number, doc.id, rank, score))
                    rank += 1
                
                print(query_number)
                query_number += 1
                
                
if __name__ == '__main__':
    ir = IRSystem('files' + os.path.sep + 'Trec_microblog11.txt')
    
    #bert re-rank
    ir.run_query(
        'files' + os.path.sep + 'topics_MB1-49.txt', 'output.txt',
        K = 1000, eval = True, method = 'bm-25', refine = True, rerank = True
                 )
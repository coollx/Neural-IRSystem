# Neural-IRSystem
Information retrivial system based on neural networks

Document.py: Store a document object, one document is represented by a vector
IRSystem.py: Our main program
QueryParser.py: Query iterator, read query file and generate query text
Tokenizer.py: preprocessing utility
Models: folder to store the word2vec model
Output: folder to store output results.
Files (To run our program, you should create all the folders by your own)
  |-StopWords.txt: File that stores stop words
  |-topics_MB1-49.txt: File that stores queries
  |-Trec_microblog11.txt: File that contains all documents to be indexed


We implemented a neural information retrieval system which indexes a collection of Twitter posts, then we tested our system on 49 queries: we take the top 1000 result which match query the best. 
Our system takes one document file (i.e. Trec_microblog11.txt) which is the union of every documents that we want to index and a query file (i.e. topics_MB1-49.txt) which contains some query that we are interested in then produce a TREC-format output file, the output file contains top 1000 matches of each query.
To run the program, just enter ‘python Neural-IRSystem.py’ in the parent directory.

Jiaxun Gao & Xiang Li
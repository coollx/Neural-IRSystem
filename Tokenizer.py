import nltk
import string
import re
#nltk.download('punkt')
#nltk.download('stopwords')
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


class Tokenizer:
    stopwords = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    #stemmer = PorterStemmer()
    lemma = nltk.wordnet.WordNetLemmatizer()
    tweet_tokenizer = TweetTokenizer()
    
    def get_tokens(self, text : str):
        tokens = self.tweet_tokenizer.tokenize(text)
        temp = []
    
        for token in tokens:
            if not token in self.stopwords and \
                not 'http' in token and \
                not 'www' in token and \
                not token in string.punctuation and \
                not '@' in token:
                if token[0] == '#':
                    temp.append(token[1:])
                else:
                    temp.append(token)
        
        ret = []
        for token in temp:
            ret.extend(token.split('-'))
         
        return [self.stemmer.stem(t) for t in ret if len(t) > 1]
        #return [self.stemmer.stem(self.lemma.lemmatize(t).lower()) for t in ret]


if __name__ == '__main__':
    tkn = Tokenizer()
    print(tkn.get_tokens('Hello , ,,,             \n  world "I am tired! I like fruit...and milk '))
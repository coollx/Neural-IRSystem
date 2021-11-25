import nltk
import string
import re
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class Tokenizer:
    stopwords = stopwords.words('english')
    stemmer = PorterStemmer()
    
    def remove_punc(self, line):
        '''
        str -> str, remove punctuations from a given string
        '''
        return line.translate(str.maketrans(' ', ' ', string.punctuation))
    
    def remove_url(self, line):
        return re.sub(r"(?:\@|https?\://)\S+", '', line)

    def get_tokens(self, text : str):
        ret = []
        text = text.lower() #convert to lower case
        text = self.remove_url(text) #remove url
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) #replace punctuation by white spaces
        text = ' '.join(text.split()) #remove multiple spaces
        text = text.replace('-', ' ') #replace hyphen by space
        
        words = word_tokenize(text)
        for word in words:
            if word not in self.stopwords: #stopwords removal
                s = self.stemmer.stem(word)
                if len(s) > 1:
                    ret.append(s)
        return ret


if __name__ == '__main__':
    tkn = Tokenizer()
    print(tkn.get_tokens3('Hello , ,,,\n  world "I am tired! I like fruit...and milk '))
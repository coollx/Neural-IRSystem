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
    
    def remove_url(self, line):
        return re.sub(r'http\S+', '', line)

    def get_tokens(self, text : str):
        ret = []
        text = text.lower() #convert to lower case
        text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) #replace punctuation by white spaces
        text = ' '.join(text.split()) #remove multiple spaces
        text = text.replace('-', ' ') #replace hyphen by space
        text = self.remove_url(text) #remove url
            
        words = word_tokenize(text)
        for word in words:
            if word not in self.stopwords: #stopwords removal
                s = self.stemmer.stem(word)
                ret.append(s)
        return ret
    

if __name__ == '__main__':
    tkn = Tokenizer()
    print(tkn.get_tokens('Hello , ,,,\n  world "I am tired! I like fruit...and milk '))
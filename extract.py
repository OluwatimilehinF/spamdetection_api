import numpy as np
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
ps = PorterStemmer()

#process email's content
import re
import string


def process_email(content) :
    '''
    preprocesses the content of an email 
    
    and returns a dictionary with word as key and its frequency as value
    @content : email content (a string)
    @return : a counting dictionary 
    '''                                         
    if not isinstance(content,str) :       
        return {},''
    content = re.sub(r'<[^<>]+>', ' ',content)  ##strip all HTML
    content = str.lower(content) ##lower case
    
    #handle URLS with http:// or https://
    content = re.sub(r'(http|https)://[^\s]*','httpaddr ',content) 
    
    #handle email addresses
    #look for strings with @ in the middle
    content = re.sub(r'[^\s]+@[^\s]+','emailaddr',content)
    
    content = re.sub(r'[0-9]+', 'number ',content) #handle numbers
    content = re.sub(r'[$]+','dollar ',content) #handle $ sign 
    content = re.sub(r'[\n]+',' ',content) #remove \n
    #remove punctuaion
    content = re.sub(r'[{0}]'.format(string.punctuation),' ',content) 
    
    res = {}
    words = word_tokenize(content)
    content = ' '.join([ps.stem(word) for word in words])
    for word in words :
        word = ps.stem(word)
        if len(word) > 11 :
            continue
        if len(word) <=1 :
            continue
        if not res.get(word):
            res[word] = 0
        res[word] += 1
    
    return res,content


class emailToFeature:
    '''
    This is a class for building feature vectors
    '''
    def __init__(self,filename) :
        vocab = pd.read_csv(filename)
        vocab = list(vocab['words'])
        index = 0
        vocabulary = {}
        while index < len(vocab) :
            vocabulary[vocab[index]] = index
            index+=1
        self.d = len(vocab)
        self.vocab = vocabulary   
    
    def fea_vector(self,email) :
        '''
        return a numpy array(1Xn) representing the
        feature vector
        @email: input email can be both a string and email object
        '''
        if type(email) is str:
            judge = email
        else :
            judge = email.get_payload()
        if not type(judge) is str:
            dic_email = {}
            for e in judge :
                dic_toadd = process_email(e.get_payload())
                for word in dic_toadd[0] :
                    if not dic_email.get(word):
                        dic_email[word] = 0
                    dic_email[word] += 1
        else :
            dic_email = process_email(judge)[0]
            
        res = np.zeros((1,self.d))
        for word in dic_email.keys() :
            if not self.vocab.get(word):
                continue
            index = self.vocab[word]
            res[0,index] = 1
        return res
    
    def build_vectors(self,is_spam,emails) :
        '''
        build feature vectors
        
        @emails : list of ham or spam emails
        @return : numpy array representing feature vectors
        '''
        N = len(emails)  # N*d array
        fea_vectors = np.zeros((N,self.d+1))
        for i in range(N) :
            a = self.fea_vector(emails[i])
            fea_vectors[i,:-1] = a
        if is_spam :
            fea_vectors[:,self.d] = 1
        else :
            fea_vectors[:,self.d] = 0
        return fea_vectors       
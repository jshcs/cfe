import tensorflow as tf
import numpy as np
from config import *
from umass_parser import *
from features_tokens import *
import pickle
import time
from gensim.models.keyedvectors import KeyedVectors
#import simstring
from sklearn.utils import shuffle

# with open(VOCAB_JNAMES,'rb') as v:
#     all_vocab=pickle.load(v)
# with open(BIO_SRT,'rb') as v:
#     all_bio_vocab=pickle.load(v)
# print "all_bio_vocab loaded...."
# WV=KeyedVectors.load_word2vec_format(WE_BIN, binary=True)
# print "embeddings loaded...."
# sorted_fname=read_sorted_file_into_array(SORTED_FPERSON_FNAME)
# print "sorted_fperson_name loaded...."
# sorted_lname=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
# print "sorted_lperson_name_loaded...."
# bio_dict={voc:1 for voc in all_bio_vocab}
# journal_dict={voc:1 for voc in all_vocab}

#sorted_journals=read_sorted_file_into_array(COMBINED_JNAMES)
# sorted_journals_db=simstring.reader(DB_JNAMES)
# print "journal_name_db loaded...."
# sorted_journals_db.measure=SS_METRIC
# sorted_journals_db.threshold=SS_THRESHOLD
# sorted_journals=[[t.lower() for t in ele.split()] for ele in sorted_journals]
#print sorted_journals
# coding: utf-8

# In[26]:
# from readDataset import read_bibtex_dataset
import readDataset
import numpy as np


styleFile = ['tfnlm','sageH']

#input the name of style file, such as 'biochem'
#output the train dictionary, val dictionary, test dictionary
#dictionary[citation string][0] = list of tokens
#dictionary[citation string][1] = index of label
dataPath = '../../data/'


    


# In[30]:


def get_raw_data(style):
    biblabels={'author':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}
    pairs = np.load(dataPath+style+'.npy')
    
    testData = {}
    
    strings = np.arange(pairs.shape[0])    
    num = pairs.shape[0]
    
    #strings[i] is the i
    #pairs[i][j][0] is the jth token of ith string
    #pairs[i][j][1] is the jth label of ith string
    for i in range(0,pairs.shape[0]):
        tokenList = []
        labelList = []
        for j in range(len(pairs[i])):
            tokenList.append(pairs[i][j][0])
            if pairs[i][j][1] in biblabels:
                labelList.append(biblabels[pairs[i][j][1]])
            else:
                labelList.append(6)
        testData[strings[i]] = (tokenList,labelList)
        
    return testData


# In[33]:


#for s in styleFile:
testData = get_raw_data('tfnlm')
x1,y1=readDataset.read_bibtex_dataset(testData)
testData=get_raw_data('sageH')
x2,y2=readDataset.read_bibtex_dataset(testData)
x1,y1=np.concatenate((x1,x2),axis=0),np.concatenate((y1,y2),axis=0)


print x1.shape,y1.shape
x1,y1=shuffle(x1,y1,random_state=0)

np.savez_compressed('../../data/we_npy_no_bio/final_test.npz',X_test=x1,y_test=y1)


# In[32]:

#
# for i in range(5):
#     print testData[i]


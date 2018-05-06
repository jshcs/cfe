
# coding: utf-8

# In[26]:


import numpy as np

Bibtexlabels = {'author':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}
styleFile = ['tfnlm','sageH']

#input the name of style file, such as 'biochem'
#output the train dictionary, val dictionary, test dictionary
#dictionary[citation string][0] = list of tokens
#dictionary[citation string][1] = index of label
dataPath = '../../data/'


    


# In[30]:


def dict_of_style(style):
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
            if pairs[i][j][1] in Bibtexlabels:
                labelList.append(Bibtexlabels[pairs[i][j][1]])
            else:
                labelList.append(12)
        testData[strings[i]] = (tokenList,labelList)
        
    return testData


# In[33]:


for s in styleFile:
    testData = dict_of_style(s)
    print len(testData)


# In[32]:


for i in range(5):
    print testData[i]



# coding: utf-8

# In[68]:


import numpy as np

styleFile = ['biochem','bmc-mathphys','mit-chicago','natbib','siamplain','spbasic','vancouver']

Bibtexlabels = {'author':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}

def dict_of_style(style):
    strings = np.load('final_'+style+'_strings.npy')
    pairs = np.load('final_'+style+'_pairs.npy')
    
    data = {}
    
    #strings[i] is the ith string
    #pairs[i][j][0] is the jth token of ith string
    #pairs[i][j][1] is the jth label of ith string
    for i in range(strings.shape[0]):
        tokenList = []
        labelList = []
        for j in range(len(pairs[i])):
            tokenList.append(pairs[i][j][0])
            if pairs[i][j][1] in labels:
                labelList.append(labels[pairs[i][j][1]])
            else:
                labelList.append(6)
        data[strings[i]] = (tokenList,labelList)
        
    return data




# In[69]:


d = dict_of_style(styleFile[0])

num = 0

for s in d:
    print s
    print d[s][0]
    print d[s][1]
    num = num+1
    if num==5:
        break

        


# In[70]:


haha = np.load('final_biochem_pairs.npy')

print haha.shape
#haha.shape[0] is the number of citation strings
#haha[i][j] is the jth token and label of the ith citation string
for i in range(len(haha[0])):
    print haha[0][i][0],haha[4][0][1]
#haha[i][j][0] is the jth token of the ith citation string
#haha[i][j][1] is the jth label of the ith citation string


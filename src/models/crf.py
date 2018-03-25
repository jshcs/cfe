
# coding: utf-8

# In[ ]:

input = '<ref-marker> [30] </ref-marker> <authors> <person> <person-first> E. </person-first> <person-middle> W. </person-middle> <person-last> Montroll , </person-last> </person> <person> <person-first> B. </person-first> <person-middle> J. </person-middle> <person-last> West , </person-last> </person> </authors> <venue> <booktitle> Fluctuation Phenomena , </booktitle> <publisher> Elsevier Science Publishers B. V. , </publisher> <address> Amsterdam , </address> <date> <year> 1979 , </year> </date> <chapter> Ch . On an enriched collection of stochastic processes , </chapter> <pages> pp . 61--205 . </pages> </venue> '


# In[4]:

import lxml.etree
import xml.etree.ElementTree as ET
import csv

parser = lxml.etree.XMLParser(recover = True)
spl = {'&' : 'Rand' , '"' : 'Rdquote' , "'" :  'Rquote'}

labels = [] 
inputs = [[]]
output = [[]]
S =[]
s_dict = {}
with open('dev.xml','r') as f :
    content = f.readlines()
for lines in content[:1000]:
    sentence = []
    temp_dict = {}
    aline = '<NODE>' + lines + '</NODE>'
  #  print aline
    for splchar in spl.keys():
        if splchar in aline :
            aline = aline.replace(splchar, spl[splchar])
    tree_rec = ET.fromstring(aline,parser)
    for ele in tree_rec.iter() :
        if ele .tag != 'NODE':
            temp_dict[ele.text] = ele.tag
            sentence.append(ele.text)
   # print temp_dict
    print sentence
    S.append(sentence)
  #  print sentence ,'\n' , temp_dict
    s_dict[' '.join(sentence)] = temp_dict
    labels.append(temp_dict.values())
#print "HELLO\n", S,"\nHELLO\n", s_dict    
print labels[0]


# In[5]:

'''
doc = [] 
for sen in S :
    temp_doc = []
    s = ' '.join(sen)
    dict = s_dict[s]
  #  print "\nAYAY\n", ' '.join(dict.keys())
    for keys in dict.keys() :
        use_B = True
        for words in list(keys.split(' ')) :
            if words not in ' ' :
                temp_doc.append((words,dict[keys]))
    doc.append(temp_doc)
print doc[0] ,'\n\n\n' , doc[1]
              
'''
doc = [] 
for sen in S :
    temp_doc = []
    s = ' '.join(sen)
    dict = s_dict[s]
    t = []
    for s in sen :
        if s not in ' ':
            t.append(s)
    for keys in t:
        for words in list(keys.split(' ')) :
            if words not in ' ' :
                temp_doc.append((words,dict[keys]))
    doc.append(temp_doc)
print doc[0] ,'\n\n\n' , doc[1]


# In[6]:

def features(doc,i) :
    word = doc[i][0]
    tag = doc[i][1]
    
    features = [
        'bias' ,
        'word.lower=' + word.lower(),
        'word.isdigit=%s' %word.isdigit(),
        'word.istitle=%s' %word.istitle()
    ]
    #if tag is person names 
    features.append(tag)
    return features

features(doc[0],5)
    


# In[7]:


import datetime

#Feature names
feature_names=[
"is_all_caps",
"is_capitalized",
"is_alpha_num",
"word_length",
"is_number",
"ends_with_period",
"enclosed_brackets",
"has_hyphen",
"has_colon",
"is_etal",
"is_valid_year",
"is_special_token",
#"first_name_lexicon",
#"last_name_lexicon",
]

def binary_search(arr,s,start,end):
	if start<=end:
		mid=int((start+end)/2)
		if arr[mid]==s:
			return True
		elif arr[mid]<s:
			start=mid+1
			return binary_search(arr,s,start,end)
		else:
			end=mid-1
			return binary_search(arr,s,start,end)
	else:
		return False

BRACKETS={'(':')','[':']','{':'}'}

SPCL_KEYS=['Page', 'Pg.', 'Vol.', 'Volume', 'page', 'pg.', 'vol.', 'volume']

class Features():
	def __init__(self,token):
		self.token=token[0]
		self.features={k:None for k in feature_names}

	def is_all_caps(self):
		self.features["is_all_caps"]=self.token.isupper()

	def is_capitalized(self):
		self.features["is_capitalized"]=self.token[0].isupper()

	def is_alpha_num(self):
		self.features["is_alpha_num"]=self.token.isalnum()

	def word_length(self):
		self.features["word_length"]=len(self.token)

	def is_number(self):
		self.features["is_number"]=self.token.isdigit()

	def ends_with_period(self):
		self.features["ends_with_period"]=self.token[-1]=='.'

	def enclosed_brackets(self):
		if self.token[0] in BRACKETS:
			if self.token[-1]==BRACKETS[self.token[0]]:
				self.features["enclosed_brackets"]=True
			else:
				self.features["enclosed_brackets"]=False
		else:
			self.features["enclosed_brackets"]=False

	def has_hyphen(self):
		self.features["has_hyphen"]=binary_search(sorted(self.token),'-',0,len(self.token)-1)

	def has_colon(self):
		self.features["has_colon"]=binary_search(sorted(self.token),':',0,len(self.token)-1)

	def is_etal(self):
		self.features["is_etal"]=self.token=='et' or self.token=='al'

	def is_valid_year(self):
		self.features["is_valid_year"]=self.features["is_number"] and self.features["word_length"]<=4 and 1<=int(self.token)<=datetime.datetime.now().year

	def is_special_token(self):
		self.features["is_special_token"]=binary_search(SPCL_KEYS,self.token,0,len(SPCL_KEYS)-1) 
        '''

	def first_name_lexicon(self):
		arr=read_sorted_file_into_array(SORTED_FPERSON_FNAME)
		start=0
		end=len(arr)-1
		self.features["first_name_lexicon"]=binary_search(arr,self.token.upper(),start,end)

	def last_name_lexicon(self):
		arr=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
		start=0
		end=len(arr)-1
		self.features["last_name_lexicon"]=binary_search(arr,self.token.upper(),start,end) '''

	def get_features(self):
		self.is_all_caps()
		self.is_capitalized()
		self.is_alpha_num()
		self.word_length()
		self.is_number()
		self.ends_with_period()
		self.enclosed_brackets()
		self.has_hyphen()
		self.has_colon()
		self.is_etal()
		self.is_valid_year()
		self.is_special_token()
		#self.first_name_lexicon()
		#self.last_name_lexicon()
		return self.features

def convert_dict(features) :
    alist = []
    for a,b in features.items():
        alist.append('%s='%a +'%s'%b)
    return alist


# In[8]:

from sklearn.model_selection import train_test_split

# A function for extracting features in documents
def extract_features(a_doc):
    feats = []
    print a_doc
    for index in range(len(a_doc)) :
        feat = Features(a_doc[index])
        dict_of_feats = feat.get_features()
        feats.append(convert_dict(dict_of_feats))
    return feats

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, label) in doc]

X = [extract_features(a_doc) for a_doc in doc]
y = [get_labels(a_doc) for a_doc in doc]

print X[0][0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:

import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 1000,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')


# In[10]:

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
i = 0
for x, y in zip(y_pred[i],y_test[i]):
    print x , y


# In[11]:

import numpy as np
from sklearn.metrics import classification_report

# Print out the classification report
for a_test, a_true in zip(y_test,y_pred):
    print classification_report(a_test,a_true)
    print "\n\n"
    #print(classification_report(y_test, y_pred)


# In[15]:

from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[16]:

get_ipython().run_cell_magic(u'time', u'', u"crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs', \n    c1=0.1, \n    c2=0.1, \n    max_iterations=100, \n    all_possible_transitions=True\n)\ncrf.fit(X_train, y_train)\nlabels = list(crf.classes_)\nlabels\ny_pred = crf.predict(X_test)\nmetrics.flat_f1_score(y_test, y_pred, \n                      average='weighted', labels=labels)\n# group B and I results\nsorted_labels = sorted(\n    labels, \n    key=lambda name: (name[1:], name[0])\n)\nprint(metrics.flat_classification_report(\n    y_test, y_pred, labels=sorted_labels, digits=3\n))")


# In[14]:

get_ipython().run_cell_magic(u'time', u'', u"# define fixed parameters and parameters to search\ncrf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs', \n    max_iterations=100, \n    all_possible_transitions=True\n)\nparams_space = {\n    'c1': scipy.stats.expon(scale=0.5),\n    'c2': scipy.stats.expon(scale=0.05),\n}\n\n# use the same metric for evaluation\nf1_scorer = make_scorer(metrics.flat_f1_score, \n                        average='weighted', labels = labels)\n\n# search\nrs = RandomizedSearchCV(crf, params_space, \n                        cv=3, \n                        verbose=1, \n                        n_jobs=-1, \n                        n_iter=50, \n                        scoring=f1_scorer)\nrs.fit(X_train, y_train)\n# crf = rs.best_estimator_\nprint('best params:', rs.best_params_)\nprint('best CV score:', rs.best_score_)\nprint('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))\n_x = [s.parameters['c1'] for s in rs.grid_scores_]\n_y = [s.parameters['c2'] for s in rs.grid_scores_]\n_c = [s.mean_validation_score for s in rs.grid_scores_]\n")


# In[ ]:

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
    min(_c), max(_c)
))

ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))
crf = rs.best_estimator_
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))


# In[ ]:

from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])


# In[ ]:

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])


# In[ ]:




# In[ ]:




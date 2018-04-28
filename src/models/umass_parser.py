import lxml.etree
import xml.etree.ElementTree as ET
import csv
from config import *
#from utils import read_file_into_array,read_sorted_file_into_array
import random
import pickle

def read_sorted_file_into_array(filename):
    res=[]
    f=open(filename,'r')
    for line in f:
        if line[:-1]!="":
            res.append(line[:-1])
        #print line[:-1]
    return res

def read_file_into_array(filename):
    res=[]
    f=open(filename,'r')
    for line in f:
        if line[:-1]!="":
            res.append(line[:-1])
        #print line[:-1]
    return list(set(res))

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

class GetDict():
    def __init__(self,filename):
        self.filename = filename
        with open(filename,'r') as f:
            self.content=f.readlines()
        self.file_dict= {TRAIN_FILE : '../../data/umass_train_data', TEST_FILE : '../../data/umass_test_data' , DEV_FILE :'../../data/umass_val_data'}
        self.labels=[]
        self.citation_strings=[]
        self.token_label={}
        self.parser=lxml.etree.XMLParser(recover = True)
        self.spl={'&':'Rand','"':'Rdquote',"'":'Rquote'}
        self.journal_names=read_sorted_file_into_array(SORTED_JNAMES)
        self.bio_titles=read_file_into_array(BIO_TITLES)
        # to test if year is not proper
        self.count_year_labels = []

    def save_data(self, filename, X, Y , save_dict, labels):
        z = raw_input("Do you want to overwrite the existing pickle files ? Y/N")
        if z == 'Y' :
            z = raw_input("Are u sure?")
        if z!= 'Y':
            print "not overwriting"
            return
        fname = self.file_dict[filename]
        with open(fname + '.pkl', 'wb') as outp:
            pickle.dump(X, outp)
            pickle.dump(Y, outp)

        with open(fname +'_dict.pkl', 'wb') as handle :
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fname +'_labels_dict.pkl', 'wb') as handle :
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_all_saved_dict(self):
        fname = self.file_dict[self.filename]

        with open(fname + '.pkl', 'rb') as intp :
            X = pickle.load(intp)
            Y = pickle.load(intp)

        with open(fname + '_dict.pkl', 'rb') as handle:
            all_dicts = pickle.load(handle)

        with open(fname +'_labels_dict.pkl', 'rb') as handle :
            labels = pickle.load(handle)

        return X,Y , all_dicts , labels


    def make_dict(self):
        X = []
        Y = []
        for line in self.content:
            curr_sentence=[]
            temp_dict={}
            tmp_labels=[]
            mod_string='<NODE>'+line+'</NODE>'
            for spchar in self.spl.keys():
                if spchar in mod_string:
                    mod_string=mod_string.replace(spchar,self.spl[spchar])
            x = []
            y = []
            tree_rec=ET.fromstring(mod_string,self.parser)
            for ele in tree_rec.iter():
                if ele.text!=" ":
                    if ele.tag!='NODE':
                        if 'person' in ele.tag:
                            txt=ele.text.strip().split(" ")
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                temp_dict[t]='person'
                                tmp_labels.append(labels['person'])
                                x.append(t)
                                y.append('author')
                        elif 'journal' in ele.tag:
                            txt=random.choice(self.journal_names).strip().split(" ")
                            #print txt
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                temp_dict[t]='journal'
                                tmp_labels.append(labels['journal'])
                                x.append(t)
                                y.append('journal')
                        elif 'title' in ele.tag:
                            txt=random.choice(self.bio_titles).strip().split(" ")
                            #print txt
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                temp_dict[t]='title'
                                tmp_labels.append(labels['title'])
                                x.append(t)
                                y.append('title')
                        else:
                            txt=ele.text.strip().split(" ")
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                #if t not in PUNCT:
                                temp_dict[t]=ele.tag
                                if ele.tag not in labels:
                                    labels[ele.tag] = len(labels)
                                    tmp_labels.append(labels[ele.tag])
                                    x.append(t)
                                    y.append(ele.tag)
                                else:
                                    tmp_labels.append(labels[ele.tag])
                                    x.append(t)
                                    y.append(ele.tag)
                                # else:
                                # 	print t
                        curr_sentence=curr_sentence+txt
                        #print curr_sentence
            X.append(x)
            Y.append(y)
            self.citation_strings.append(' '.join(curr_sentence))
            self.labels.append(tmp_labels)
            self.token_label[self.citation_strings[-1]]=(curr_sentence,tmp_labels)
        self.save_data(self.filename,X,Y,self.token_label, labels)


    def get_dict(self,citation_string):
        return self.token_label[citation_string]

    def get_all_dict(self):
        print len(self.token_label)
        return self.token_label

    def print_all_dict(self):
        for s in self.citation_strings:
            print "Sentence"
            print "*"*20
            print s
            print "*"*20
            print "key-values"
            print "*"*20
            print self.token_label[s]
            print
            print "*"*20

    def get_size(self):
        return len(self.citation_strings)


'''
test=GetDict(TEST_FILE)
# Comment this , it will re write all existing file
Z = test.make_dict()
X, Y, dict , tlabels = test.get_all_saved_dict()
keys = []
count = 0
print tlabels.keys()
for x,y in zip(X,Y) :
    sentence = ' '.join(x)
    x1, y1 = dict[sentence]
    for a,b,c,d in  zip(x,x1,y,y1):
        if a!=b :
            print zip(x,x1)
            break
        if c in tlabels.keys() and tlabels[c] == d :
            continue
        elif c == d :
            continue
        else :
            print 'ele' ,c , d


print len(test.count_year_labels)
print test.count_year_labels
print test.get_all_dict().items()
print 'citation'
print test.citation_strings
print test.token_label[test.citation_strings[0]][0]
print test.token_label[test.citation_strings[0]][1]
print len(test.token_label[test.citation_strings[0]][0])
print test.get_size()
'''


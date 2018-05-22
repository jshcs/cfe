import lxml.etree
import xml.etree.ElementTree as ET
import csv
from config import *
#from utils import read_file_into_array,read_sorted_file_into_array
import random

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
    def __init__(self,filename,sentence_file):

        with open(filename,'r') as f:
            self.content=f.readlines()
        self.labels=[]
        self.citation_strings=[]
        self.token_label={}
        self.parser=lxml.etree.XMLParser(recover = True)
        self.spl={'&':'Rand','"':'Rdquote',"'":'Rquote'}
        self.journal_names=read_sorted_file_into_array(SORTED_JNAMES)
        self.bio_titles=read_file_into_array(BIO_TITLES)
        self.sentence_file=sentence_file
    def make_dict(self):
        for line in self.content:
            curr_sentence=[]
            temp_dict={}
            tmp_labels=[]
            mod_string='<NODE>'+line+'</NODE>'
            for spchar in self.spl.keys():
                if spchar in mod_string:
                    mod_string=mod_string.replace(spchar,self.spl[spchar])
            tree_rec=ET.fromstring(mod_string,self.parser)
            for ele in tree_rec.iter():
                if ele.text!=" ":
                    if ele.tag!='NODE':
                        if 'person' in ele.tag:
                            txt=ele.text.strip().split(" ")
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            #print txt
                            for t in txt:
                                #if t not in PUNCT:
                                temp_dict[t]='person'
                                tmp_labels.append(labels['person'])
                                # else:
                                # 	print t
                        elif 'journal' in ele.tag:
                            txt=random.choice(self.journal_names).strip().split(" ")
                            #print txt
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                temp_dict[t]='journal'
                                tmp_labels.append(labels['journal'])
                        elif 'title' in ele.tag:
                            txt=random.choice(self.bio_titles).strip().split(" ")
                            #print txt
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                temp_dict[t]='title'
                                tmp_labels.append(labels['title'])
                        else:
                            txt=ele.text.strip().split(" ")
                            txt=[t for t in txt if t not in PUNCT and len(t)>=1 and is_ascii(t)]
                            for t in txt:
                                #if t not in PUNCT:
                                temp_dict[t]=ele.tag
                                if ele.tag not in labels:
                                    tmp_labels.append(len(labels))
                                else:
                                    tmp_labels.append(labels[ele.tag])
                                # else:
                                # 	print t
                        curr_sentence=curr_sentence+txt
                        #print curr_sentence

            self.citation_strings.append(' '.join(curr_sentence))
            self.labels.append(tmp_labels)
            self.token_label[self.citation_strings[-1]]=(curr_sentence,tmp_labels)
        np.save(self.sentence_file,self.citation_strings,allow_pickle=True)

    def get_dict(self,citation_string):
        return self.token_label[citation_string]
    def get_all_dict(self):
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




##test=GetDict(TRAIN_FILE)
##test.make_dict()
##test.print_all_dict()
##print test.citation_strings[0]
##print test.token_label[test.citation_strings[0]][0]
##print test.token_label[test.citation_strings[0]][1]
##print len(test.token_label[test.citation_strings[0]][0])
##print test.get_size()

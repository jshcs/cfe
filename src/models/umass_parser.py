import lxml.etree
import xml.etree.ElementTree as ET
import csv
from config import *

class GetDict():
	def __init__(self,filename):
		with open(filename,'r') as f:
			self.content=f.readlines()
		self.labels=[]
		self.citation_strings=[]
		self.token_label={}
		self.parser=lxml.etree.XMLParser(recover = True)
		self.spl={'&':'Rand','"':'Rdquote',"'":'Rquote'}
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
							txt=[t for t in txt if t not in PUNCT]
							#print txt
							for t in txt:
								#if t not in PUNCT:
								temp_dict[t]='person'
								tmp_labels.append('person')
								# else:
								# 	print t
						else:
							txt=ele.text.strip().split(" ")
							txt=[t for t in txt if t not in PUNCT]
							for t in txt:
								#if t not in PUNCT:
								temp_dict[t]=ele.tag
								tmp_labels.append(ele.tag)
								# else:
								# 	print t
						curr_sentence=curr_sentence+txt
						#print curr_sentence

			self.citation_strings.append(' '.join(curr_sentence))
			self.labels.append(tmp_labels)
			self.token_label[self.citation_strings[-1]]=(curr_sentence,tmp_labels)

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

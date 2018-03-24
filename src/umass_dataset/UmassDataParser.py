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
			mod_string='<NODE>'+line+'</NODE>'
			for spchar in self.spl.keys():
				if spchar in mod_string:
					mod_string=mod_string.replace(spchar,self.spl[spchar])
			tree_rec=ET.fromstring(mod_string,self.parser)
			for ele in tree_rec.iter():
				if ele.text!=" ":
					if ele.tag!='NODE':
						if 'person' in ele.tag:
							temp_dict[ele.text.strip()]='person'
						else:
							temp_dict[ele.text.strip()]=ele.tag
						curr_sentence.append(ele.text.strip())
			self.citation_strings.append(' '.join(curr_sentence))
			self.token_label[self.citation_strings[-1]]=temp_dict
			self.labels.append(temp_dict.values())
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


# test=GetDict(TEST_FILE)
# test.make_dict()
# test.print_all_dict()
# print test.get_size()
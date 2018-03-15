from config import *
from utils import *

class Features():
	def __init__(self,token):
		self.token=token
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

	def first_name_lexicon(self):
		arr=read_sorted_file_into_array(SORTED_FPERSON_FNAME)
		start=0
		end=len(arr)-1
		self.features["first_name_lexicon"]=binary_search(arr,self.token.upper(),start,end)

	def last_name_lexicon(self):
		arr=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
		start=0
		end=len(arr)-1
		self.features["last_name_lexicon"]=binary_search(arr,self.token.upper(),start,end)

	def get_features(self):
		self.is_all_caps()
		self.is_capitalized()
		self.is_alpha_num()
		self.word_length()
		self.is_number()
		self.ends_with_period()
		self.first_name_lexicon()
		self.last_name_lexicon()
		return self.features



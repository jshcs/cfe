from config import *
from utils import *
import datetime
import numpy as np

class Features():
	def __init__(self,token):
		self.token=token
		self.features={k:None for k in config_params['feature_names']}

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
		self.enclosed_brackets()
		self.has_hyphen()
		self.has_colon()
		self.is_etal()
		self.is_valid_year()
		self.is_special_token()
		self.first_name_lexicon()
		self.last_name_lexicon()
		return self.features

	def vectorize(self):
		fDict = self.get_features()
		v = np.array(fDict.values())

		return v


# add to dir path
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1,dir_path +'/../')
print sys.path

from config import *

from utils import *
import datetime
import pickle

indir_vocab_jnames = '../' + VOCAB_JNAMES
indir_bio_srt = '../' + BIO_SRT
indir_sorted_fperson_fname = '../'+SORTED_FPERSON_FNAME
indir_sorted_lperson_fname = '../'+SORTED_LPERSON_FNAME


with open(indir_vocab_jnames,'rb') as v:
    all_vocab=pickle.load(v)

with open(indir_bio_srt,'rb') as read:
    all_bio_vocab=pickle.load(read)

all_bio_vocab = [a.decode('utf-8') for a in all_bio_vocab]

sorted_fname= read_sorted_file_into_array(indir_sorted_fperson_fname)
sorted_lname= read_sorted_file_into_array(indir_sorted_lperson_fname)
MAX_WINDOW=5
DB_JNAMES="../../../data/journal_db/db-journals.db"


class CRF_Features():

    def __init__(self,token):
        self.token = token
        self.jnames_vocab=all_vocab
        self.bioterms_vocab=all_bio_vocab
        self.features = {k:False for k in config_params['feature_names']}

    def is_title(self):
        self.features["is_title"] = self.token.istitle()

    def is_upper(self):
        self.features["is_upper"] = self.token.isupper()

    def is_alpha_num(self):
        self.features["is_alpha_num"] = self.token.isalnum()

    def word_length(self):
        self.features["word_length"] = len(self.token)

    def is_num(self):
        self.features["is_number"] = self.token.isdigit()

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
        self.features["has_hyphen"] = False
        parts = self.token.split('-')
        if len(parts) > 1 :
            self.features["has_hyphen"] = True
            is_digit = True
            for part in parts :
                if part != '':
                    is_digit = is_digit and part.isdigit()
            self.features["is_number"] = is_digit

    def has_colon(self):
        self.features["has_colon"]= False
        parts = self.token.split(':')
        if len(parts) > 1 :
            self.features["has_colon"]= True

    def is_etal(self):
        self.features["et_al"] = self.token == 'et' or self.token == 'al'

    def is_valid_year(self):
        self.features["is_valid_year"] = self.token.isdigit() and self.features["word_length"] <= 4 \
        and self.features["word_length"] >=2 and 1<=int(self.token)<=datetime.datetime.now().year

    def is_special_token(self):
        self.features["is_special_token"] = True if self.token in SPCL_KEYS else False

    def has_period_period(self): #12
        # s=time.time()
        if ".." in self.token:
            self.features["has_period_period"]=True

    def has_period_comma(self): #13
        if ".," in self.token:
            self.features["has_period_comma"]=True

    def is_url(self): #14
        if "http://" in self.token or "www." in self.token :
                self.features["is_url"]=True

    def is_email(self): #15
        stra =  self.token
        if '@' in stra and '.' in stra.split('@')[1] :
            self.features["is_email"]=True

    def first_name_lexicon(self): #16
        # s=time.time()
        if len(self.token)==2 and self.features["is_upper"] and self.features["ends_with_period"]:
            self.features["first_name_lexicon"]=True
            return
        arr= sorted_fname
        start=0
        end=len(arr)-1
        self.features["first_name_lexicon"]=binary_search(arr,self.token.upper(),start,end)

    def last_name_lexicon(self): #17
        # s=time.time()
        #arr=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
        arr= sorted_lname
        start=0
        end=len(arr)-1
        self.features["last_name_lexicon"]=binary_search(arr,self.token.upper(),start,end)
        # e=time.time()
        # self.times.append(e-s)

    def journal_lexicon(self): #18
        '''
        if binary_search(all_vocab,self.token.lower(),0,len(all_vocab)-1):
            self.features["journal_lexicon"]=True
        else:
            for w in all_vocab:
                if len(w)>=len(self.token):
                    if float(longest_common_substring(self.token.lower(),w))/max(len(self.token),len(w))>=0.6:
                        self.features["journal_lexicon"]=True
                        break
        '''
        if self.token.lower() in self.jnames_vocab:
            self.features['journal_name']=True


    def is_bio_term(self): #19
        token = self.token.decode('utf-8')
        self.features["is_bio_term"]=binary_search(all_bio_vocab,token.lower(),0,len(all_bio_vocab)-1)


    def get_features(self):
        self.is_title()
        self.is_upper()
        self.is_alpha_num()
        self.word_length()
        self.is_num()
        self.ends_with_period()
        self.enclosed_brackets()
        self.has_hyphen()
        self.has_colon()
        self.is_etal()
        self.is_valid_year()
        self.is_special_token()
        self.has_period_comma()
        self.has_period_period()
        self.is_url()
        self.is_email()
        self.first_name_lexicon()
        self.last_name_lexicon()
        self.is_bio_term()
        self.journal_lexicon()
        return self.features
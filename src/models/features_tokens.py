from config import *
from utils import *
import datetime
import numpy as np
import validators

class Features():
    def __init__(self,sentence,vocab_journals,vocab_bioterms):
        #self.token=token
        self.jnames_vocab=vocab_journals
        self.bioterms_vocab=vocab_bioterms
        #self.features={k:False for k in config_params['feature_names']}
        self.features=[]
        self.sentence=sentence
        for tok in self.sentence:
            self.features.append([False for i in range(len(config_params['feature_names']))])


    def is_all_caps(self,token): #0
        #return token.isupper()
        self.features[self.sentence.index(token)][0]=token.isupper()

    def is_capitalized(self,token): #1
        self.features[self.sentence.index(token)][1]=token[0].isupper()
        #return token[0].isupper()

    def is_alpha_num(self,token): #2
        self.features[self.sentence.index(token)][2]=token.isalnum()
        #return token.isalnum()

    def word_length(self,token): #3
        self.features[self.sentence.index(token)][3]=len(token)
        #return len(token)

    def is_number(self,token): #4
        self.features[self.sentence.index(token)][4]=token.isdigit()
        #return token.isdigit()

    def ends_with_period(self,token): #5
        self.features[self.sentence.index(token)][5]=token[-1]=='.'
        #return token[-1]=="."

    def enclosed_brackets(self,token): #6
        if token[0] in BRACKETS:
            if token[-1]==BRACKETS[token[0]]:
                self.features[self.sentence.index(token)][6]=True
            else:
                self.features[self.sentence.index(token)][6]=False
        else:
            self.features[self.sentence.index(token)][6]=False
        # if token[0] in BRACKETS:
        #     if token[-1]==BRACKETS[token[0]]:
        #         return True
        #     else:
        #         return False
        # else:
        #     return False




    def has_hyphen(self,token): #7
        self.features[self.sentence.index(token)][7]=binary_search(sorted(token),'-',0,len(token)-1)
        #return binary_search(sorted(token),"-",0,len(token)-1)

    def has_colon(self,token): #8
        self.features[self.sentence.index(token)][8]=binary_search(sorted(token),':',0,len(token)-1)
        #return binary_search(sorted(token),':',0,len(token)-1)

    def is_etal(self,token): #9
        self.features[self.sentence.index(token)][9]=token=='et' or token=='al'
        #return token=='et' or token=='al'

    def is_valid_year(self,token): #10
        self.features[self.sentence.index(token)][10]=self.features[self.sentence.index(token)][4] and self.features[self.sentence.index(token)][3]<=4 and 1<=int(token)<=datetime.datetime.now().year or ((token[0]=='`' or token[0]=="'") and self.is_number(token[1:]) and self.features[self.sentence.index(token)][3]==3 and 1<=int(token[1:])<=datetime.datetime.now().year)
        #return (self.is_number(token) and self.word_length(token)<=4 and 1<=int(token)<=datetime.datetime.now().year) or ((token[0]=='`' or token[0]=="'") and self.word_length(token)==3 and 1<=int(token[1:])<=datetime.datetime.now().year)

    def is_special_token(self,token): #11
        self.features[self.sentence.index(token)][11]=binary_search(SPCL_KEYS,token,0,len(SPCL_KEYS)-1)
        #return binary_search(SPCL_KEYS,token,0,len(SPCL_KEYS)-1)

    def has_period_period(self,token): #12
        if ".." in token:
            self.features[self.sentence.index(token)][12]=True
        # if ".." in token:
        #     return True

    def has_period_comma(self,token): #13
        if ".," in token:
            self.features[self.sentence.index(token)][13]=True


    def is_url(self,token): #14
        if validators.url(token):
            self.features[self.sentence.index(token)][14]=True

    def is_email(self,token): #15
        if validators.email(token):
            self.features[self.sentence.index(token)][15]=True

    def first_name_lexicon(self,token): #16
        if len(token)==2 and self.features[self.sentence.index(token)][1] and self.features[self.sentence.index(token)][5]:
            self.features[self.sentence.index(token)][16]=True
            return
        arr=read_sorted_file_into_array(SORTED_FPERSON_FNAME)
        start=0
        end=len(arr)-1
        self.features[self.sentence.index(token)][16]=binary_search(arr,token.upper(),start,end)

    def last_name_lexicon(self,token): #17
        arr=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
        start=0
        end=len(arr)-1
        self.features[self.sentence.index(token)][17]=binary_search(arr,token.upper(),start,end)

    def journal_lexicon(self,token): #18
        if token.lower() in self.jnames_vocab:
            self.features[self.sentence.index(token)][18]=True
        else:
            for w in self.jnames_vocab:
                if len(w)>=len(token):
                    if float(longest_common_substring(token,w))/max(len(token),len(w))>=0.6:
                        self.features[self.sentence.index(token)][18]=True
                        break

    def is_bio_term(self,token): #19
        if token.lower() in self.bioterms_vocab:
            self.features[self.sentence.index(token)][19]=True
        else:
            for w in self.bioterms_vocab:
                if len(w)>=len(token):
                    if float(longest_common_substring(token,w))/max(len(token),len(w))>=0.6:
                        self.features[self.sentence.index(token)][19]=True
                        break

    def get_features(self):
        for tok in self.sentence:
            self.is_all_caps(tok)
            self.is_capitalized(tok)
            self.is_alpha_num(tok)
            self.word_length(tok)
            self.is_number(tok)
            self.ends_with_period(tok)
            self.enclosed_brackets(tok)
            self.has_hyphen(tok)
            self.has_colon(tok)
            self.is_etal(tok)
            self.is_valid_year(tok)
            self.is_special_token(tok)
            self.has_period_period(tok)
            self.has_period_comma(tok)
            self.is_url(tok)
            self.is_email(tok)
            self.first_name_lexicon(tok)
            self.last_name_lexicon(tok)
            self.journal_lexicon(tok)
            self.is_bio_term(tok)
        return np.array(self.features)

    def vectorize(self):
        #fDict = self.get_features()


        v = np.array(self.features)

        return v


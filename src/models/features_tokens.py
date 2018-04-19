from config import *
from utils import *
import datetime
import numpy as np
import validators
import time

# def Features(sentence,fname_list,lname_list,vocab_journals,vocab_bioterms):
#     features=[]
#     for tok in sentence:
#         features.append([False for i in range(len(config_params['feature_names']))])
#
#     for token in range(len(sentence)):
#         if sentence[token]=='<UNK>':
#             continue
#         else:
#             #is_all_caps
#             features[token][0]=sentence[token].isupper()
#             #is_capitalized
#             features[token][1]=sentence[token][0].isupper()
#             #is_alpha_num
#             features[token][2]=sentence[token].isalnum()
#             #word_length
#             features[token][3]=len(sentence[token])
#             #is_number
#             features[token][4]=sentence[token].isdigit()
#             #ends_with_period
#             features[token][5]=sentence[token][-1]=='.'
#             #enclosed_brackets
#             if sentence[token][0] in BRACKETS:
#                 if sentence[token][-1]==BRACKETS[sentence[token][0]]:
#                     features[token][6]=True
#                 else:
#                     features[token][6]=False
#             else:
#                 features[token][6]=False
#             #has_hyphen
#             features[token][7]=binary_search(sorted(sentence[token]),'-',0,len(sentence[token])-1)
#
#             #has_colon
#             features[token][8]=binary_search(sorted(sentence[token]),':',0,len(sentence[token])-1)
#             #is_etal
#             features[token][9]=sentence[token]=='et' or sentence[token]=='al'
#             #is_valid_year
#             features[token][10]=features[token][4] and features[token][3]<=4 and 1<=int(sentence[token])<=datetime.datetime.now().year or ((sentence[token][0]=='`' or sentence[token][0]=="'") and sentence[token][1:].isdigit() and features[token][3]==3 and 1<=int(sentence[token][1:])<=datetime.datetime.now().year)
#             #is_special_token
#             features[token][11]=binary_search(SPCL_KEYS,sentence[token],0,len(SPCL_KEYS)-1)
#             #has_period_period
#             if ".." in sentence[token]:
#                 features[token][12]=True
#             #has_period_comma
#             if ".," in sentence[token]:
#                 features[token][13]=True
#             #is_url
#             if validators.url(sentence[token]):
#                 features[token][14]=True
#             #is_email
#             if validators.email(sentence[token]):
#                 features[token][15]=True
#             #first_name_lexicon
#             if len(sentence[token])==2 and features[token][1] and features[token][5]:
#                 features[token][16]=True
#                 # e=time.time()
#                 # self.times.append(e-s)
#             else:
#                 start=0
#                 end=len(fname_list)-1
#                 features[token][16]=binary_search(fname_list,sentence[token].upper(),start,end)
#             #last_name_lexicon
#             start=0
#             end=len(lname_list)-1
#             features[token][17]=binary_search(lname_list,sentence[token].upper(),start,end)
#
#             #journal_lexicon
#             if binary_search(vocab_journals,sentence[token].lower(),0,len(vocab_journals)-1):
#                 features[token][18]=True
#             else:
#                 for w in vocab_journals:
#                     if len(w)>=len(sentence[token]):
#                         if float(longest_common_substring(sentence[token].lower(),w))/max(len(sentence[token]),len(w))>=0.6:
#                             features[token][18]=True
#                             break
#
#
#             #is_bio_term
#             features[token][19]=binary_search(vocab_bioterms,sentence[token].lower(),0,len(vocab_bioterms)-1)
#     return features



class Features():
    def __init__(self,sentence,fname_list,lname_list,vocab_journals,vocab_bioterms,sorted_journals):
        #self.token=token
        self.jnames_vocab=vocab_journals
        self.bioterms_vocab=vocab_bioterms
        len_sorted_journals=len(sorted_journals)
        self.sorted_journals_1=sorted_journals
        #self.sorted_journals_2=sorted_journals[len_sorted_journals/2:]
        #self.sorted_journals_3=sorted_journals[2*len_sorted_journals/3:]
        #self.features={k:False for k in config_params['feature_names']}
        self.features=[]
        self.sentence=sentence
        for tok in self.sentence:
            self.features.append([False for i in range(len(config_params['feature_names']))])
        self.fname_list=fname_list
        self.lname_list=lname_list
        #self.times=[]


    def is_all_caps(self,token): #0
        #return token.isupper()
        #s=time.time()
        self.features[token][0]=self.sentence[token].isupper()
        # e=time.time()
        # self.times.append(e-s)

    def is_capitalized(self,token): #1
        # s=time.time()
        self.features[token][1]=self.sentence[token][0].isupper()
        # e=time.time()
        # self.times.append(e-s)
        #return token[0].isupper()

    def is_alpha_num(self,token): #2
        # s=time.time()
        self.features[token][2]=self.sentence[token].isalnum()
        # e=time.time()
        # self.times.append(e-s)
        #return token.isalnum()

    def word_length(self,token): #3
        # s=time.time()
        self.features[token][3]=len(self.sentence[token])
        # e=time.time()
        # self.times.append(e-s)
        #return len(token)

    def is_number(self,token): #4
        # s=time.time()
        self.features[token][4]=self.sentence[token].isdigit()
        # e=time.time()
        # self.times.append(e-s)
        #return token.isdigit()

    def ends_with_period(self,token): #5
        # s=time.time()
        self.features[token][5]=self.sentence[token][-1]=='.'
        # e=time.time()
        # self.times.append(e-s)
        #return token[-1]=="."

    def enclosed_brackets(self,token): #6
        # s=time.time()
        if self.sentence[token][0] in BRACKETS:
            if self.sentence[token][-1]==BRACKETS[self.sentence[token][0]]:
                self.features[token][6]=True
            else:
                self.features[token][6]=False
        else:
            self.features[token][6]=False
        # e=time.time()
        # self.times.append(e-s)
        # if token[0] in BRACKETS:
        #     if token[-1]==BRACKETS[token[0]]:
        #         return True
        #     else:
        #         return False
        # else:
        #     return False




    def has_hyphen(self,token): #7
        # s=time.time()
        self.features[token][7]=binary_search(sorted(self.sentence[token]),'-',0,len(self.sentence[token])-1)
        # e=time.time()
        # self.times.append(e-s)
        #return binary_search(sorted(token),"-",0,len(token)-1)

    def has_colon(self,token): #8
        # s=time.time()
        self.features[token][8]=binary_search(sorted(self.sentence[token]),':',0,len(self.sentence[token])-1)
        # e=time.time()
        # self.times.append(e-s)
        #return binary_search(sorted(token),':',0,len(token)-1)

    def is_etal(self,token): #9
        # s=time.time()
        self.features[token][9]=self.sentence[token]=='et' or self.sentence[token]=='al'
        # e=time.time()
        # self.times.append(e-s)
        #return token=='et' or token=='al'

    def is_valid_year(self,token): #10
        # s=time.time()
        self.features[token][10]=self.features[token][4] and self.features[token][3]<=4 and 1<=int(self.sentence[token])<=datetime.datetime.now().year or ((self.sentence[token][0]=='`' or self.sentence[token][0]=="'") and self.sentence[token][1:].isdigit() and self.features[token][3]==3 and 1<=int(self.sentence[token][1:])<=datetime.datetime.now().year)
        # e=time.time()
        # self.times.append(e-s)
        #return (self.is_number(token) and self.word_length(token)<=4 and 1<=int(token)<=datetime.datetime.now().year) or ((token[0]=='`' or token[0]=="'") and self.word_length(token)==3 and 1<=int(token[1:])<=datetime.datetime.now().year)

    def is_special_token(self,token): #11
        # s=time.time()
        self.features[token][11]=binary_search(SPCL_KEYS,self.sentence[token],0,len(SPCL_KEYS)-1)
        # e=time.time()
        # self.times.append(e-s)
        #return binary_search(SPCL_KEYS,token,0,len(SPCL_KEYS)-1)

    def has_period_period(self,token): #12
        # s=time.time()
        if ".." in self.sentence[token]:
            self.features[token][12]=True
        # e=time.time()
        # self.times.append(e-s)
        # if ".." in token:
        #     return True

    def has_period_comma(self,token): #13
        # s=time.time()
        if ".," in self.sentence[token]:
            self.features[token][13]=True

        # # e=time.time()
        # self.times.append(e-s)


    def is_url(self,token): #14
        # s=time.time()
        if validators.url(self.sentence[token]):
            self.features[token][14]=True
        # e=time.time()
        # self.times.append(e-s)

    def is_email(self,token): #15
        # s=time.time()
        if validators.email(self.sentence[token]):
            self.features[token][15]=True
        # e=time.time()
        # self.times.append(e-s)

    def first_name_lexicon(self,token): #16
        # s=time.time()
        if len(self.sentence[token])==2 and self.features[token][1] and self.features[token][5]:
            self.features[token][16]=True
            # e=time.time()
            # self.times.append(e-s)
            return
        arr=self.fname_list
        start=0
        end=len(arr)-1
        self.features[token][16]=binary_search(arr,self.sentence[token].upper(),start,end)
        # e=time.time()
        # self.times.append(e-s)

    def last_name_lexicon(self,token): #17
        # s=time.time()
        #arr=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
        arr=self.lname_list
        start=0
        end=len(arr)-1
        self.features[token][17]=binary_search(arr,self.sentence[token].upper(),start,end)
        # e=time.time()
        # self.times.append(e-s)

    # def journal_lexicon(self,token): #18
    #     # s=time.time()
    #     # if binary_search(self.jnames_vocab,self.sentence[token].lower(),0,len(self.jnames_vocab)-1):
    #     if self.sentence[token].lower() in self.jnames_vocab:
    #         self.features[token][18]=True
    #     else:
    #         for w in self.jnames_vocab.keys():
    #             if len(w)>=len(self.sentence[token]):
    #                 if float(longest_common_substring(self.sentence[token].lower(),w))/max(len(self.sentence[token]),len(w))>=0.6:
    #                     self.features[token][18]=True
    #                     break
        # e=time.time()
        # self.times.append(e-s)

    def journal_lexicon(self,token): #18
        present=False
        upper=token
        for win in range(1,MAX_WINDOW):
            if token+win+1<=len(self.sentence):
                ss=[s.lower() for s in self.sentence[token:token+win+1] if s!="<UNK>"]
                substr=' '.join(ss)
                present=binary_search_with_fuzzing(self.sorted_journals_1,str(substr),0,len(self.sorted_journals_1)-1,0.5)
                if present==True:
                    print "****",substr
                    upper=win+token
        if present:
            #print token,upper
            if upper+1<=len(self.sentence):
                for i in range(token,upper+1):
                    self.features[i][18]=True
            else:
                upper-=1
                for i in range(token,upper+1):
                    self.features[i][18]=True

    def is_bio_term(self,token): #19
        #s=time.time()
        # if self.sentence[token].lower() in self.bioterms_vocab:
        #     self.features[token][19]=True
        # else:
        # for w in self.bioterms_vocab:
        #     if len(w)>=len(token):
        #         if float(longest_common_substring(token.lower(),w))/max(len(token),len(w))>=0.6:
        #             self.features[self.sentence.index(token)][19]=True
        #             break
        # e=time.time()
        # self.times.append(e-s)
        #self.features[token][19]=binary_search(self.bioterms_vocab,self.sentence[token].lower(),0,len(self.bioterms_vocab)-1)
        self.features[token][19]=self.sentence[token].lower() in self.bioterms_vocab


    def get_features(self):
        e=[0 for i in range(len(config_params["feature_names"]))]
        c=0
        for tok in range(len(self.sentence)):
            if self.sentence[tok]=='<UNK>':
                continue
            else:
                c+=1
                s=time.time()
                self.is_all_caps(tok)
                e[0]+=(time.time()-s)
                self.is_capitalized(tok)
                e[1]+=(time.time()-s)
                self.is_alpha_num(tok)
                e[2]+=(time.time()-s)
                self.word_length(tok)
                e[3]+=(time.time()-s)
                self.is_number(tok)
                e[4]+=(time.time()-s)
                self.ends_with_period(tok)
                e[5]+=(time.time()-s)
                self.enclosed_brackets(tok)
                e[6]+=(time.time()-s)
                self.has_hyphen(tok)
                e[7]+=(time.time()-s)
                self.has_colon(tok)
                e[8]+=(time.time()-s)
                self.is_etal(tok)
                e[9]+=(time.time()-s)
                self.is_valid_year(tok)
                e[10]+=(time.time()-s)
                self.is_special_token(tok)
                e[11]+=(time.time()-s)
                self.has_period_period(tok)
                e[12]+=(time.time()-s)
                self.has_period_comma(tok)
                e[13]+=(time.time()-s)
                self.is_url(tok)
                e[14]+=(time.time()-s)
                self.is_email(tok)
                e[15]+=(time.time()-s)
                self.first_name_lexicon(tok)
                e[16]+=(time.time()-s)
                self.last_name_lexicon(tok)
                e[17]+=(time.time()-s)
                if self.features[tok][18]==False:
                    self.journal_lexicon(tok)
                e[18]+=(time.time()-s)
                self.is_bio_term(tok)
                e[19]+=(time.time()-s)
                # print (e1-s),(e2-s),(e3-s),(e4-s),(e5-s),(e6-s),(e7-s),(e8-s),(e9-s),(e10-s),(e11-s),(e12-s),(e13-s),(e14-s),(e15-s),(e16-s),(e17-s),(e18-s),(e19-s),(e20-s),
                # print
                # print
        print e
        return self.features

    def vectorize(self):
        #fDict = self.get_features()


        v = np.array(self.features)

        return v


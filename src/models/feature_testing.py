from features import *
from config import *
from utils import *
import pickle

with open(VOCAB_JNAMES,'rb') as v:
    all_vocab=pickle.load(v)
with open(BIO_SRT,'rb') as v:
    all_bio_vocab=pickle.load(v)

token="https://www.google.com"

t=Features(token,all_vocab,all_bio_vocab)
f=t.get_features()

print "Total features:",len(f)
print
print f
print
print "The key-value pairs are:"
print
print t.vectorize()

'''
Total features: 14

['is_valid_year', 'first_name_lexicon', 'is_alpha_num', 'is_capitalized', 'last_name_lexicon', 'is_all_caps', 'is_etal', 'has_hyphen', 'word_length', 'enclosed_brackets', 'is_number', 'ends_with_period', 'has_colon', 'is_special_token']

The key-value pairs are:

{'is_valid_year': False, 'first_name_lexicon': False, 'is_alpha_num': False, 'is_capitalized': False, 'last_name_lexicon': False, 'is_all_caps': False, 'is_etal': False, 'has_hyphen': True, 'word_length': 8, 'enclosed_brackets': True, 'is_number': False, 'ends_with_period': False, 'has_colon': False, 'is_special_token': False}

'''
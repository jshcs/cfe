from features import *

token="[1234]"

t=Features(token)
f=t.get_features()

print "Total features:",len(f)
print
print f.keys()
print
print "The key-value pairs are:"
print
print f

'''
Total features: 9

['first_name_lexicon', 'is_alpha_num', 'is_capitalized', 'last_name_lexicon', 'is_all_caps', 'word_length', 'enclosed_brackets', 'is_number', 'ends_with_period']

The key-value pairs are:

{'first_name_lexicon': False, 'is_alpha_num': False, 'is_capitalized': False, 'last_name_lexicon': False, 'is_all_caps': False, 'word_length': 6, 'enclosed_brackets': True, 'is_number': False, 'ends_with_period': False}

'''
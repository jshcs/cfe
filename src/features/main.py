from features import *

token="1234"

t=Features(token)

print t.get_features()


'''
{'first_name_lexicon': False, 'is_alpha_num': True, 'is_number': True, 'last_name_lexicon': False, 'word_length': 4, 'is_all_caps': False, 'ends_with_period': False}
'''
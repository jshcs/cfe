#Features
config_params={
"feature_names":[
"is_all_caps", #0
"is_capitalized", #1
"is_alpha_num", #2
"word_length", #3
"is_number", #4
"ends_with_period", #5
"enclosed_brackets", #6
"has_hyphen", #7
"has_colon", #8
"is_etal", #9
"is_valid_year", #10
"is_special_token", #11
"has_period_period", #12
"has_period_comma", #13
"is_url", #14
"is_email", #15
"first_name_lexicon", #16
"last_name_lexicon", #17
"journal_name_lexicon", #18
"is_bio_term" #19
],
#Training params
"epochs":120,
"lrate":0.00075,
"lrate_decay":0.85,
"do_bnorm":True,
"do_dropout":True,
"max_stream_length":50,
"num_units":128,
"batch_size":20,
"num_layer":1,
}

TRAIN_FILE='../../data/umass_cfe/training.docs'
TEST_FILE='../../data/umass_cfe/testing.docs'
DEV_FILE='../../data/umass_cfe/dev.docs'

SORTED_FPERSON_FNAME="../../data/sorted-person-first.txt"
SORTED_LPERSON_FNAME="../../data/sorted-person-last.txt"
UNSRT_FPERSON_FNAME="../../data/person-first.txt"
UNSRT_LPERSON_FNAME="../../data/person-last.txt"

RAW_JNAMES="../../data/biomed_journals_unsrt.txt"
UNSRT_JNAMES="../../data/journals-unsrt.txt"
SORTED_JNAMES="../../data/sorted-journals.txt"
VOCAB_JNAMES="../../data/vocab-journals.pickle"
COMBINED_JNAMES="../../data/combined-journals.txt"


RAW_BIOTITLES="../../data/vocab.txt"
BIO_SRT="../../data/bio-srt.pickle"
BIO_TITLES="../../data/bio_titles.txt"

ALL_TAGS=['person','title','year','journal','volume','pages']

PUNCT=[".",",",";",":"]

labels = {'person':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}

BRACKETS={'(':')','[':']','{':'}'}

SPCL_KEYS=['Page', 'Pg.', 'Vol.', 'Volume', 'page', 'pg.', 'vol.', 'volume']

MAX_WINDOW=5
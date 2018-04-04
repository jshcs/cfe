#Features
config_params={
"feature_names":[
"is_all_caps",
"is_capitalized",
"is_alpha_num",
"word_length",
"is_number",
"ends_with_period",
"enclosed_brackets",
"has_hyphen",
"has_colon",
"is_etal",
"is_valid_year",
"is_special_token",
"first_name_lexicon",
"last_name_lexicon",
"journal_name_lexicon",
"is_bio_term",
"has_period_period",
"has_period_comma",
"is_url",
"is_email"
],
#Training params
"epochs":100,
"lrate":5e-4,
"lrate_decay":0.7,
"do_bnorm":True,
"do_dropout":True,
"max_stream_length":115,
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

RAW_BIOTITLES="../../data/bio_titles.txt"
BIO_SRT="../../data/bio-srt.pickle"

ALL_TAGS=['person','title','year','journal','volume','pages']

PUNCT=[".",",",";",":"]

labels = {'person':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}

BRACKETS={'(':')','[':']','{':'}'}

SPCL_KEYS=['Page', 'Pg.', 'Vol.', 'Volume', 'page', 'pg.', 'vol.', 'volume']

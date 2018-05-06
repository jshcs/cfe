#Features
import simstring
import numpy as np
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
"is_bio_term", #19
"word_embeddings" #20
],
#Training params
"epochs":10,
"lrate":1e-4,
"lrate_decay":0.8,
"do_bnorm":True,
"do_dropout":True,
"max_stream_length":110,
"num_units":128,
"batch_size":100,
"num_layer":1,
"filter_width":3,
"repeat_times":4
}

TRAIN_FILE='../../data/umass_cfe/training.docs'
TEST_FILE='../../data/umass_cfe/testing.docs'
DEV_FILE='../../data/umass_cfe/dev.docs'

# TR_SENT_FILE='../../data/umass_tr_sentences.npy'
# TST_SENT_FILE='../../data/umass_tst_sentences.npy'
# DEV_SENT_FILE='../../data/umass_dev_sentences.npy'

TR_SENT_FILE='../../data/umass_tr_sentences_no_bio.npy'
TST_SENT_FILE='../../data/umass_tst_sentences_no_bio.npy'
DEV_SENT_FILE='../../data/umass_dev_sentences_no_bio.npy'


SORTED_FPERSON_FNAME="../../data/sorted-person-first.txt"
SORTED_LPERSON_FNAME="../../data/sorted-person-last.txt"
UNSRT_FPERSON_FNAME="../../data/person-first.txt"
UNSRT_LPERSON_FNAME="../../data/person-last.txt"

RAW_JNAMES="../../data/biomed_journals_unsrt.txt"
UNSRT_JNAMES="../../data/journals-unsrt.txt"
SORTED_JNAMES="../../data/sorted-journals.txt"
VOCAB_JNAMES="../../data/vocab-journals.pickle"
COMBINED_JNAMES="../../data/combined-journals.txt"

DB_JNAMES="../../data/journal_db/db-journals.db"


RAW_BIOTITLES="../../data/vocab.txt"
BIO_SRT="../../data/bio-srt.pickle"
BIO_TITLES="../../data/bio_titles.txt"


WE_BIN="../../data/PMC-w2v.bin"
WE_TXT="../../data/PMC-w2v.txt"
WE_PKL="../../data/PMC-w2v.pickle"


RESULTS="../../data/results.json"
PARAMS="../../data/params.json"


ALL_TAGS=['person','title','journal','year','volume','pages']
# ALL_TAGS=['B-person','I-person','B-title','I-title','B-journal','I-journal','B-year','I-year','B-volume','I-volume','B-pages','I-pages']


styleFile = ['biochem','bmc-mathphys','achemso','ajhg','mit-chicago','natbib','siamplain','spbasic']
#styleFile = ['mit-chicago','natbib','siamplain','spbasic']
PUNCT=[".",",",";",":"]

labels = {'person':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}
# pos_labels=['person','title','journal','year','volume','pages']
# labels={'B-person':0,'I-person':1,'B-title':2,'I-title':3,'B-journal':4,'I-journal':5,'B-year':6,'I-year':7,'B-volume':8,'I-volume':9,'B-pages':10,'I-pages':11}

BRACKETS={'(':')','[':']','{':'}'}

SPCL_KEYS=['Page', 'Pg.', 'Vol.', 'Volume', 'page', 'pg.', 'vol.', 'volume']

feature_descriptions=[
    'Is the token in uppercase?',
    'Is the token capitalized?',
    'Is the token alpha-numeric?',
    'Length of the token',
    'Is the token numeric?',
    'Does the token end with a period?',
    'Is the token enclosed within brackets?',
    'Does the token contain hyphen(s)?',
    'Does the token contain colon(s)?',
    'Is the token "et al"?',
    'Is the token a valid year?',
    'Is the token a special keyword like pg., vol., etc?',
    'Does the token end with ".."?',
    'Does the token end with ".,"?',
    'Is the token a url?',
    'Is the token an email address?',
    'Could the token be a first name?',
    'Could the token be a last name?',
    'Could the token be a journal name?',
    'Is the token a biomed term?',
    'Biomed word embedding(length 200)'

]

MODELS=['CRF','LSTM','Bi-LSTM','ID-CNN']

MAX_WINDOW=5

SS_METRIC=simstring.jaccard
SS_THRESHOLD=0.4

EMD_SIZE=200

# LR_RANGE=[1e-4,5e-4,1e-3,5e-3,1e-2]
# DECAY_RATE=[0.8,0.85,0.9]
LR_RANGE=[5e-3]
DECAY_RATE=[0.9]
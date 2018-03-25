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
],
#Training params
"epochs":10,
"lrate":3e-4,
"lrate_decay":0.9,
"do_bnorm":True,
"do_dropout":True,
"max_stream_length":20,
}

TRAIN_FILE='../../data/umass_cfe/training.docs'
TEST_FILE='../../data/umass_cfe/testing.docs'
DEV_FILE='../../data/umass_cfe/dev.docs'

ALL_TAGS=['person','title','year','journal','volume','pages']

PUNCT=[".",",",";",":"]
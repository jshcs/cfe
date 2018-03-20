SORTED_FPERSON_FNAME="../../data/sorted-person-first.txt"
SORTED_LPERSON_FNAME="../../data/sorted-person-last.txt"
UNSRT_FPERSON_FNAME="../../data/person-first.txt"
UNSRT_LPERSON_FNAME="../../data/person-last.txt"

#Feature names
feature_names=[
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
]

BRACKETS={'(':')','[':']','{':'}'}

SPCL_KEYS=['Page', 'Pg.', 'Vol.', 'Volume', 'page', 'pg.', 'vol.', 'volume']
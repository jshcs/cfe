def is_all_caps(token):
    return token.isupper()

def is_alpha_num(token):
    return token.isalnum()

def word_length(token):
    return len(token)

def is_number(token):
    return token.isdigit()

def ends_with_period(token):
    return token[-1]=='.'

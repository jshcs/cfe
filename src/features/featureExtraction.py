
def is_all_caps(token):
    for t in token:
        if t>'Z' or t<'A':
            return False

    return True

def is_alpha_num(token):
    for t in token:
        if (t<='z' and t>='a') or (t<='Z' and t>='A') or (t<='9' and t>='0'):
            continue
        else:
            return False

    return True

def word_length(token):
    return len(token)

def is_number(token):
    for t in token:
        if t>'9' or t<'0':
            return False

    return True

def ends_with_period(token):
    return token[-1]=='.'

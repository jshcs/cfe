delimiter = [',','(',')']

def sepStrings(string,delimiter):
        res = []
        temp = ''
        for s in string:
                if s in delimiter and temp!='':
                        res.append(temp)
                        temp = ''
                elif temp!='' or s!=' ':
                        temp = temp + s
                        
        if temp not in res:
                res.append(temp)
                temp = ''
        return res

def posAuthors(seg,dictionary):
        name = dictionary['author']
        authors = []
        temp = ''
        wordSep = [' ',',']
        for ch in name:
                if ch in wordSep and temp!='' and temp not in wordSep:
                        authors.append(temp)
                        temp = ''
                elif temp!='' or ch not in wordSep:
                        temp = temp+ch
        if temp not in authors and temp!='' and temp not in wordSep:
                authors.append(temp)
                temp = ''

        print(authors)
        print('author above')
        pos = []
        for s in seg:
                words=[]
                temp = ''
                for ch in s:
                        if ch in wordSep and temp!='' and temp not in wordSep:
                                words.append(temp)
                                temp = ''
                        elif temp!='' or ch not in wordSep:
                                temp = temp+ch
                if temp not in words and temp!='' and temp not in wordSep:
                        words.append(temp)
                        temp = ''
                same=0.0
                for w in words:
                        if w in authors:
                                same = same + 1
                print(words,same)
                pos.append(same/len(words))

        return pos

def posTitle(seg,dictionary):
        titles = dictionary['title']
        title = []
        temp = ''
        wordSep = [' ',',']
        for ch in titles:
                if ch in wordSep and temp!='' and temp not in wordSep:
                        title.append(temp)
                        temp = ''
                elif temp!='' or ch not in wordSep:
                        temp = temp+ch
        if temp not in title and temp!='' and temp not in wordSep:
                title.append(temp)
                temp = ''

        print(title)
        print('title above')
        pos = []
        for s in seg:
                words=[]
                temp = ''
                for ch in s:
                        if ch in wordSep and temp!='' and temp not in wordSep:
                                words.append(temp)
                                temp = ''
                        elif temp!='' or ch not in wordSep:
                                temp = temp+ch
                if temp not in words and temp!='' and temp not in wordSep:
                        words.append(temp)
                        temp = ''
                same=0.0
                for w in words:
                        if w in title:
                                same = same + 1
                print(words,same)
                pos.append(same/len(words))

        return pos
        
cstr = 'Ahuja, N., On detection and representation of multiscale low-level image structure. ACM Computing Surveys 27, 3 (Sept. 1995), 304--306.,  ACM Computing Surveys 27, 3 (Sept. 1995), 304--306.'

s = sepStrings(cstr,delimiter)
print(s)
d = {'author':'Ahuja, N.','title':'On detection and representation of multiscale low-level image structure'}
a = posAuthors(s,d)
print(a)
t = posTitle(s,d)
print(t)

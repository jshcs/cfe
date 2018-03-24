import lxml.etree
import xml.etree.ElementTree as ET
import csv

parser = lxml.etree.XMLParser(recover = True)
spl = {'&' : 'Rand' , '"' : 'Rdquote' , "'" :  'Rquote'}

labels = []
inputs = [[]]
output = [[]]
S =[]
s_dict = {}
with open('dev.docs','r') as f :
    content = f.readlines()
#get 2 strings
for lines in content[:2]:
    sentence = []
    temp_dict = {}
    aline = '<NODE>' + lines + '</NODE>'
    print aline
    for splchar in spl.keys():
        if splchar in aline :
            aline = aline.replace(splchar, spl[splchar])
    tree_rec = ET.fromstring(aline,parser)
    for ele in tree_rec.iter() :
        print ele.tag
        if ele .tag != 'NODE':
            if 'person' in ele.tag :
                temp_dict[ele.text] = 'person'
            else :
                temp_dict[ele.text] = ele.tag
            sentence.append(ele.text)
    print temp_dict
    print sentence
    S.append(sentence)
    s_dict[' '.join(sentence)] = temp_dict
    labels.append(temp_dict.values())
print type(s_dict.keys()) , type(s_dict.values())
print s_dict.items()[0]
# the key value is the input citation string
# the value label is the dict of the same string


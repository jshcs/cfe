

#using bibtexparser

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding
from bibtexparser.bibdatabase import as_text

dict = {}

with open('bib-2.bib') as bibtex_file:
    parser = BibTexParser(common_strings = True)
    parser.customization = homogenize_latex_encoding
    bib_database = bibtexparser.load(bibtex_file, parser=parser)
    for entry in range(len(bib_database.entries)) :
        listofkeys = bib_database.entries[entry].keys()
        for key in listofkeys :
            akey = as_text(key)
            dict[akey] = bib_database.entries[entry][akey]
            print akey , dict[akey]
        print '\n'

'''
#incomplete :P
#map bbl to dict

pairs = {}
pairs['Asuru:1992:OAS'] = 'cite'
match = open('Asuru:1992:OAS.bbl', 'r') 
lines = match.read()
splits =  'bibitem'
lines = lines.split('bibitem')[1]
lines = ''.join(lines.split('{'))
lines = ''.join(lines.split('}'))
lines = ''.join(lines.split('\\textit'))
lines = lines.split('\n')
print lines

for line in lines :
    print line
    wrds = line.split(' ')
    total = len(wrds)
    count = 0
    for (key, value) in dict.viewitems():
        print value
        count = 0
        for wrd in wrds :
            if wrd in value :
                count += 1
        if ''.join(wrds) in value :
            print 'total match' , ''.join(wrds) , value
'''
#manually

name = 'Asuru:1992:OAS'
bibf = open(name +'.bib' , 'r')
segments  = bibf.read().split('@article')
dictitems = segments[1].split(',\n')
name = dictitems[0][1:]
dictitems = dictitems[1:-1]
prevkey = ''
pair = {}
for items in dictitems :
    print 'n*' , items , '\n'
    if '=' in items :
        key,value = items.split('=')
        pair[key] = value
        print key , value , '\n'
        prevkey = key
    else :
        pair[prevkey] += items
        print prevkey , pair[prevkey] ,'\n'

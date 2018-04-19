import numpy as np

Bibtexlabels = {'author':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}

data_dir = '../../data/'

#input the name of style file, such as 'biochem'
#output the train dictionary, val dictionary, test dictionary
#dictionary[citation string][0] = list of tokens
#dictionary[citation string][1] = index of label
def dict_of_style(style):
    strings = np.load(data_dir+'final_'+style+'_strings.npy')
    pairs = np.load(data_dir+'final_'+style+'_pairs.npy')
    
    trainData = {}
    valData = {}
    testData = {}
    
    num = strings.shape[0]
    trainNum = int(num*0.6)
    valNum = int(num*0.25)
    testNum = num - trainNum - valNum
    
    #strings[i] is the ith string
    #pairs[i][j][0] is the jth token of ith string
    #pairs[i][j][1] is the jth label of ith string
    for i in range(0,trainNum):
        tokenList = []
        labelList = []
        for j in range(len(pairs[i])):
            tokenList.append(pairs[i][j][0])
            if pairs[i][j][1] in Bibtexlabels:
                labelList.append(Bibtexlabels[pairs[i][j][1]])
            else:
                labelList.append(6)
        trainData[strings[i]] = (tokenList,labelList)
        
    for i in range(trainNum,valNum+trainNum):
        tokenList = []
        labelList = []
        for j in range(len(pairs[i])):
            tokenList.append(pairs[i][j][0])
            if pairs[i][j][1] in Bibtexlabels:
                labelList.append(labels[Bibtexlabels[i][j][1]])
            else:
                labelList.append(6)
        valData[strings[i]] = (tokenList,labelList)
        
    for i in range(valNum+trainNum,num):
        tokenList = []
        labelList = []
        for j in range(len(pairs[i])):
            tokenList.append(pairs[i][j][0])
            if pairs[i][j][1] in Bibtexlabels:
                labelList.append(labels[Bibtexlabels[i][j][1]])
            else:
                labelList.append(6)
        testData[strings[i]] = (tokenList,labelList)
        
    return trainData,valData,testData



##trainDict,valDict,testDict = dict_of_style(styleFile[0])
##
##num = 0
##
##print len(trainDict)
##print len(valDict)
##print len(testDict)
##
##print 'train dict'
##for s in trainDict:
##    print s
##    print d[s][0]
##    print d[s][1]
##    num = num+1
##    if num==5:
##        break
##        
##print 'val dict'
##for s in valDict:
##    print s
##    print d[s][0]
##    print d[s][1]
##    num = num+1
##    if num==5:
##        break
##
##print 'test dict'
##for s in testDict:
##    print s
##    print d[s][0]
##    print d[s][1]
##    num = num+1
##    if num==5:
##        break
##
##
##
##
##haha = np.load('final_biochem_pairs.npy')
##
##print haha.shape
###haha.shape[0] is the number of citation strings
###haha[i][j] is the jth token and label of the ith citation string
##for i in range(len(haha[0])):
##    print haha[0][i][0],haha[4][0][1]
###haha[i][j][0] is the jth token of the ith citation string
###haha[i][j][1] is the jth label of the ith citation string
##

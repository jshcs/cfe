from gensim.models.keyedvectors import KeyedVectors
from config import *
import pickle

def save_as_txt():
    model = KeyedVectors.load_word2vec_format(WE_BIN, binary=True)
    print "Loaded...."
    model.save_word2vec_format(WE_TXT, binary=False)
    print "Saved...."

# save_as_txt()

def store_in_dict():
    f=open(WE_TXT,'r')
    we_vectors=[]
    c=0
    words=[]
    for line in f:
        tmp_arr=line.split()
        words.append(tmp_arr[0])
        we_vectors.append(tmp_arr[1:101])
        # we_vectors[tmp_arr[0]]=tmp_arr[1:]
        # print len(tmp_arr[1:])
        c+=1
        print c
        #print len(tmp_arr[1:]),c
    print "Dictionary generation done...."
    print len(we_vectors)
    # with open(WE_PKL,'wb') as out:
    #     pickle.dump(we_vectors,out)
    # print "Writing to pickle file done...."

# store_in_dict()
import time

start=time.time()
word_vectors = KeyedVectors.load_word2vec_format(WE_BIN, binary=True)
# with open(WE_PKL,'wb') as out:
#     pickle.dump(word_vectors,out)

# with open(WE_PKL, 'rb') as inp:
#     word_vectors = pickle.load(inp)

print "Loading done..."
end=time.time()
# print end-start
# start=time.time()

end=time.time()
print end-start
print word_vectors['nature']
# print word_vectors.most_similar(positive=['woman', 'king'], negative=['man'],topn=1)

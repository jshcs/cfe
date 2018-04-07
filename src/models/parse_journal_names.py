from config import *
from utils import *
import pickle

f=open(RAW_JNAMES,'r')
f_unsrt=open(UNSRT_JNAMES,'w')
f_srt=open(SORTED_JNAMES,'r')
bio_raw=open(RAW_BIOTITLES,'r')
bio_srt=open(BIO_SRT,'w')


def parse_raw():
    for line in f:
        if "JournalTitle" in line or "MedAbbr" in line or "IsoAbbr" in line:
            get_line=line.split(":")
            f_unsrt.write(get_line[1].lower())

def get_vocab(infile,outfile):
    vocab=[]
    f=open(infile,'r')
    for line in f:
        vocab+=line.split(" ")
    vocab=list(set([x.strip("\n") for x in vocab]))

    vocab.sort()
    print len(vocab)
    #print vocab
    with open(outfile,'w') as vc:
        pickle.dump(vocab,vc)


#get_vocab(f_srt,VOCAB_JNAMES)
get_vocab(RAW_BIOTITLES,BIO_SRT)
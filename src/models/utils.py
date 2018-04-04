from config import *
from umass_parser import *

def map_features_vec(features_map):
    feature_vec=[]
    for feature in config_params["feature_names"]:
        feature_vec.append(features_map[feature])
    return feature_vec

def get_data(dataset_type):
    if dataset_type=="train":
        data_obj=GetDict(TRAIN_FILE)
    elif dataset_type=="test":
        data_obj=GetDict(TEST_FILE)
    elif dataset_type=="dev":
        data_obj=GetDict(DEV_FILE)

    data_obj.make_dict()
    data=data_obj.get_all_dict()
    return data

def get_sequence(dataset_type):
    data=get_data(dataset_type)
    MAX_LEN=config_params["max_stream_length"]
    token_seq=[]
    label_seq=[]
    kv_pairs=data.values()
    for i in range(len(kv_pairs)):
        tmp=kv_pairs[i][0]
        tmp_labels=kv_pairs[i][1]
        if len(tmp)<MAX_LEN:
            tmp=tmp+["<UNK>"]*(MAX_LEN-len(tmp))
            tmp_labels=tmp_labels+["<UNK>"]*(MAX_LEN-len(tmp_labels))
        elif len(tmp)>MAX_LEN:
            tmp=tmp[:MAX_LEN]
            tmp_labels=tmp_labels[:MAX_LEN]
        tmp=[tmp[i] for i in range(len(tmp)) if tmp_labels[i] in ALL_TAGS]
        tmp_labels=[tmp_labels[i] for i in range(len(tmp_labels)) if tmp_labels[i] in ALL_TAGS]
        token_seq.append(tmp)
        label_seq.append(tmp_labels)

    return token_seq,label_seq

def binary_search(arr,s,start,end):
    if start<=end:
        mid=int((start+end)/2)
        if arr[mid]==s:
            return True
        elif arr[mid]<s:
            start=mid+1
            return binary_search(arr,s,start,end)
        else:
            end=mid-1
            return binary_search(arr,s,start,end)
    else:
        return False

def read_sorted_file_into_array(filename):
    res=[]
    f=open(filename,'r')
    for line in f:
        if line[:-1]!="":
            res.append(line[:-1])
        #print line[:-1]
    return res

def read_file_into_array(filename):
    res=[]
    f=open(filename,'r')
    for line in f:
        if line[:-1]!="":
            res.append(line[:-1])
        #print line[:-1]
    return list(set(res))

def write_array_to_file(arr,filename):
    f=open(filename,'w')
    for i in arr:
        print i
        f.write(i+"\n")
    f.close()

def sort_string_list(arr):
    arr.sort()
    return arr

def longest_common_substring(s1,s2):
    n,m=len(s1),len(s2)
    dp=[[0 for i in range(n+1)] for j in range(m+1)]
    max_len=0
    for i in range(m+1):
        for j in range(n+1):
            if i==0 or j==0:
                dp[i][j]=0
            elif s2[i-1]==s1[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
                max_len=max(max_len,dp[i][j])
            else:
                dp[i][j]=0
    return max_len









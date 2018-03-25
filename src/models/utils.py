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

print get_sequence("train")[1]









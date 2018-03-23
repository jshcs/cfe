from config import *

def map_features_vec(features_map):
	feature_vec=[]
	for feature in config_params["feature_names"]:
		feature_vec.append(features_map[feature])
	return feature_vec



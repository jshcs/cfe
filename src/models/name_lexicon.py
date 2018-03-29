from utils import *
from config import *

#input_file=SORTED_FPERSON_FNAME/SORTED_LPERSON_FNAME
def name_lexicon(input_file,s):
	arr=read_sorted_file_into_array(input_file)
	start=0
	end=len(arr)-1
	return binary_search(arr,s,start,end)


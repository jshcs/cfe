from config import *
from utils import *

#first_name_list=read_file_into_array(UNSRT_FPERSON_FNAME)
# last_name_list=read_file_into_array(UNSRT_LPERSON_FNAME)
journal_name_list=read_file_into_array(UNSRT_JNAMES)
journal_name_list2=read_sorted_file_into_array(SORTED_JNAMES)

# first_name_list_srt=sort_string_list(first_name_list)
# last_name_list_srt=sort_string_list(last_name_list)
# journal_name_list_srt=sort_string_list(journal_name_list)
#
# write_array_to_file(first_name_list_srt,SORTED_FPERSON_FNAME)
# write_array_to_file(last_name_list_srt,SORTED_LPERSON_FNAME)
# write_array_to_file(journal_name_list_srt,SORTED_JNAMES)
print len(journal_name_list2),len(journal_name_list)
combined=list(set(journal_name_list+journal_name_list2))
print len(combined)
combined.sort()

write_array_to_file(combined,COMBINED_JNAMES)

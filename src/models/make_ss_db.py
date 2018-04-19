import simstring
from config import *
from utils import *
db = simstring.writer(DB_JNAMES)

sorted_journals=read_sorted_file_into_array(COMBINED_JNAMES)

for j in sorted_journals:
    db.insert(j)

db.close()

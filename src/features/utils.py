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
			res.append(line[:-1].lower())
		#print line[:-1]
	return res

def read_file_into_array(filename):
	res=[]
	f=open(filename,'r')
	for line in f:
		if line[:-1]!="":
			res.append(line[:-1].lower())
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

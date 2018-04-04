import re
import subprocess
import os
#re.split('; |, |\*|\n',a)

bsts=['acm','ajhg','ama','bbs','bioinformatics','cbe','cell','ecology']

def compiletex(filename):
	FNULL = open(os.devnull, 'w')
	subprocess.call(["latexmk",filename+".tex"],stdout=FNULL, stderr=subprocess.STDOUT)	

def maketex(texfilename,bibname,stylefile):
	g=open(texfilename+".tex","w")
	bib=open(bibname+".bib","r")
	g.write("\\documentclass[11pt]{article}\n")
	g.write("\\usepackage{cite}\n")
	g.write("\\begin{document}\n")

	for line in bib:
		#print '*'
		if '@Article' in line or '@article' in line:
			print line
			line.replace('@Article','')
			line.replace('@article','')
			line.replace(' ','')
			line.replace(',','')
			print line
			line=line.split(',')[0][9:]
			g.write('~\cite{'+line+'}\n')

	g.write("\\bibliography{"+bibname+"}{}\n")
	g.write("\\bibliographystyle{"+stylefile+"}\n")
	g.write("\\end{document}\n")
	g.close()

#for x in bsts:
#	maketex(x,'bib-2',x)
#	compiletex(x)

def cleanstring(x):
	badstrspace=["~"]
	badstrnull=["{","}","\n","\\em","\\it","\\newblock"]
	for items in badstrspace:
		x = x.replace(items, " ")
	for items in badstrnull:
		x = x.replace(items, "")

	x=x.replace("\ldots.","...")
	x=x.replace("--","-")

	return x

def makepairs(order,dictionary):
	curdict=[]
	for x in order:
		print x," : ", dictionary[x]
		temp=re.split(" +", cleanstring(dictionary[x]))
		for y in temp:
			if y!='':
				curdict+=[[y,x]]
	return curdict


def iteratebib(filename,helper):
	stringlist=[]
	dctlist=[]

	bbl=open(filename+".bbl","r")
	#citations=open("strings"+filename+".txt","w")
	#dictfile=open("dictionary"+filename+".txt","w")

	line=bbl.readline()
	while line:
		if line[:8]=="\\bibitem":
			a,b=helper(bbl)
			stringlist+=[a]
			dctlist+=[b]
			print "+++++++++++++++++++++++++++++"
			print a			
			print "+++++++++++++++++++++++++++++"
			print b
		line=bbl.readline()

	return stringlist,dctlist

def cbehelper(bbl):
	#\bibitem[\protect\astroncite{Marriott and Stuckey}{1993}]{Marriott:1993:SCL}
	#{\sc Marriott, K. and Stuckey, P.~J.} 1993.
	#\newblock Semantics of constraint logic programs with optimization.
	#\newblock {\em ACM Letters on Programming Languages and Systems} 2:197--212.
	order=["author","year","title","journal","number","pages"]
	d={}
	ctn=""

	a2=bbl.readline()
	while(a2[:4]!="{\sc"):
		a2=bbl.readline()

	#Author line
	authorline=a2
	ctn+=authorline[5:]
	authorline=authorline[5:]
	a2=bbl.readline()
	while a2[:9]!="\\newblock":
		ctn+=a2
		authorline+=a2
		a2=bbl.readline()

	d["author"]=authorline[:-7]
	d["year"]=authorline[-6:]
	
	#title line
	title=a2[9:]
	ctn+=a2[9:]

	nxt=bbl.readline()
	while nxt[:9]!="\\newblock":
		ctn+=nxt
		title+=nxt
		nxt=bbl.readline()

	d["title"]=title
	
	#journal and pages
	journal=nxt[9:]
	ctn+=nxt[9:]

	nxt=bbl.readline()
	while nxt!="\n":
		ctn+=nxt
		title+=nxt
		nxt=bbl.readline()

	journal=journal.split("}")
	tail=journal[1]
	d["journal"]=journal[0]
	#print "tail:", tail
	tail=tail.split(":")
	d["number"]=tail[0]
	d["pages"]=tail[1]

	return cleanstring(ctn),makepairs(order,d)


def ajhghelper(bbl):
	order=["author","year","title","journal","number","pages"]
	d={}
	ctn=""	
	
	a2=bbl.readline()
	
	#Author line
	authorline=a2
	ctn+=authorline
	#authorline=authorline[5:]
	a2=bbl.readline()
	while a2[:9]!="\\newblock":
		ctn+=a2
		authorline+=a2
		a2=bbl.readline()

	author=authorline

	#Year Line
	year=a2[10:]
	ctn+=year

	#\bibitem{Choudhary:1993:UCF}
	#Choudhary, A., Fox, G., Hiranandani, S., Kennedy, K., Koelbel, C., Ranka, S.,
	#  and Tseng, C.-W.
	#\newblock (1993).
	#\newblock Unified compilation of {Fortran 77D} and {90D}.
	#\newblock ACM Letters on Programming Languages and Systems {\em 2}, 95--114.
	#title line
	a2=bbl.readline()
	title=a2[9:]
	ctn+=a2[9:]

	nxt=bbl.readline()
	while nxt[:9]!="\\newblock":
		ctn+=nxt
		title+=nxt
		nxt=bbl.readline()
	
	print "TITLE : ",title
	journal=nxt[9:]
	ctn+=nxt[9:]

	nxt=bbl.readline()
	while nxt!="\n":
		ctn+=nxt
		title+=nxt
		nxt=bbl.readline()

	journal=journal.split("{")
	print "JOURNAL : ",journal
	tail=journal[1]
	journal=journal[0]
	print "tail:", tail
	tail=tail.split(",")
	number=tail[0]
	pages=tail[1]

	d["author"]=author
	d["title"]=title
	d["journal"]=journal
	d["year"]=year
	d["number"]=number
	d["pages"]=pages

	return cleanstring(ctn),makepairs(order,d)
	

def amahelper(bbl):
	order=["author","title","journal","year","number","pages"]
	d={}
	ctn=""	
	#Nilsen Kelvin~D., Schmidt William~J.. Cost-effective object space management
	#for hardware-assisted real-time garbage collection  {\it ACM Letters on
	#Programming Languages and Systems. } 1992;1:338--354.
	a=bbl.readline()
	ctn=""
	whole=""
	curdict=[]

	while a!="\n":
		whole+=a
		ctn+=a
		a=bbl.readline()

	print "++++++++++++++++++++++++++++"
	print whole
	print "++++++++++++++++++++++++++++"
	print cleanstring(whole)

	whole=whole.replace("\\it","{{")
	whole=re.split("{{|. }",whole)

	authorplustitle=whole[0]
	yearplusvolume=whole[-1]
	whole=''.join(whole[1:-1])

	authorplustitle=authorplustitle.split(". ")
	d["author"]=authorplustitle[0]+". "
	d["title"]=authorplustitle[1]
	
	d["journal"]=whole+". "

	yearplusvolume=yearplusvolume.split(";")	
	d["year"]=yearplusvolume[0]
	volumepluspages=yearplusvolume[1]
	volumepluspages=volumepluspages.split(":")
	d["number"]=volumepluspages[0]
	d["pages"]=pages=volumepluspages[1]

	return cleanstring(ctn),makepairs(order,d)






#x="ama"
#maketex(x,'bib-2',x)
#compiletex(x)
#a,b = ama("ama")
#a,b = ajhg("ajhg")
a,b = iteratebib("ama",amahelper)
print a
for c in b:
	print "++++++++++++++++++++++++++++++++"
	print c
#c3="Ball, T. What's in a region?:or computing control dependence regions in near-linear time for reducible control flow. ACM Letters on Programming Languages and Systems 2, 4 (Mar.,  1993), 1-16."

#separate(c3)
import re
import subprocess
import os

def arstyle1helper(bbl):
        order=["author","year","title","journal","number","pages"]
        d={}
        ctn=""	

##\bibitem[Ball(1993)]{Ball:1993:WRC}
##Ball T. 1993.
##What's in a region?: or computing control dependence regions in near-linear
##  time for reducible control flow.
##\textit{ACM Letters on Programming Languages and Systems} 2:1--16

	#author and year
        authorline = ""
        a=bbl.readline()
        while a[:8]=='\\bibitem':
                a = bbl.readline()
                
        cnt+=a
        authorline+=a
        while not a[-5:-1].isdigit():
                a = bbl.readline()
                cnt+=a
                authorline+=a

        d["author"]=authorline[:-4]
        d["year"]=authorline[-5:]
	
	#title line
        a = bbl.readline()
        title = ''
        while a[:8]!='\\textit{':
                cnt+=a
                title+=a
                a = bbl.readline()

        d["title"]=title

        #journal,volume and pages
        venueline = ''
        venueline+=a[8:]
        while not a[-1].isdigit():
            venueline+=a
            a = bbl.readline()
        [journal,venue] = venueline.split('}')
        ctn+=journal
        ctn+=venue
        d["journal"] = journal
        [volume,page] = venue.split(':')
        d["number"] = volume
        d["pages"] = page

        return cleanstring(ctn),makepairs(order,d)

def arstyle3helper(bbl):
        order=["author","year","title","journal","number","pages"]
        d={}
        ctn=""	

##\bibitem{Burke:1992:PEI}
##Burke M, Choi JD. 1992.
##Precise and efficient integration of interprocedural alias information into
##  data-flow analysis.
##\textit{ACM Letters on Programming Languages and Systems} 1:14--21

	#author and year
        authorline = ""
        a=bbl.readline()
        while a[:8]=='\\bibitem':
                a = bbl.readline()
                
        cnt+=a
        authorline+=a
        while not a[-5:-1].isdigit():
                a = bbl.readline()
                cnt+=a
                authorline+=a

        d["author"]=authorline[:-4]
        d["year"]=authorline[-5:]
	
	#title line
        a = bbl.readline()
        title = ''
        while a[:8]!='\\textit{':
                cnt+=a
                title+=a
                a = bbl.readline()

        d["title"]=title

        #journal,volume and pages
        venueline = ''
        venueline+=a[8:]
        while not a[-1].isdigit():
            venueline+=a
            a = bbl.readline()
        [journal,venue] = venueline.split('}')
        ctn+=journal
        ctn+=venue
        d["journal"] = journal
        [volume,page] = venue.split(':')
        d["number"] = volume
        d["pages"] = page

        return cleanstring(ctn),makepairs(order,d)

def arstyle6helper(bbl):
        order=["author","year","title","journal","number","pages"]
        d={}
        ctn=""	

##\bibitem{Ball:1993:WRC}
##Ball T. 1993.
##What's in a region?: or computing control dependence regions in near-linear
##  time for reducible control flow.
##\textit{ACM Letters on Programming Languages and Systems} 2:1--16

	#author and year
        authorline = ""
        a=bbl.readline()
        while a[:8]=='\\bibitem':
                a = bbl.readline()
                
        cnt+=a
        authorline+=a
        while not a[-5:-1].isdigit():
                a = bbl.readline()
                cnt+=a
                authorline+=a

        d["author"]=authorline[:-4]
        d["year"]=authorline[-5:]
	
	#title line
        a = bbl.readline()
        title = ''
        while a[:8]!='\\textit{':
                cnt+=a
                title+=a
                a = bbl.readline()

        d["title"]=title

        #journal,volume and pages
        venueline = ''
        venueline+=a[8:]
        while not a[-1].isdigit():
            venueline+=a
            a = bbl.readline()
        [journal,venue] = venueline.split('}')
        ctn+=journal
        ctn+=venue
        d["journal"] = journal
        [volume,page] = venue.split(':')
        d["number"] = volume
        d["pages"] = page

        return cleanstring(ctn),makepairs(order,d)

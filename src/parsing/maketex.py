
import shlex, subprocess

z = open('samplePaper.tex','r')
lines = z.readlines()
cite_name = 'a'
lines[10] = 'Blablabla said Nobody ~\cite{{0}}.'.format(cite_name)
lines[12] = '\bibliography{{0}}{}'.format(cite_name)

texopen = open('normal.txt','r')
fnames = texopen.readlines()
texopen.close()

ret_list = []

for fname in fnames :
    tex = open(fname +'.tex' ,'w')
    cite_name = '{' + fname + '}'
    lines[10] = 'Blablabla said Nobody ~\cite{}.'.format(cite_name)
    lines[12] = '\\bibliography{}'.format(cite_name) +'{}\n'
    for line in lines :
        tex.write(line)
    tex.close()
    latex_cmd = 'latex {}'.format(fname)
    latex_args = shlex.split(latex_cmd)
    p = subprocess.Popen(latex_args)

    bib_cmd = 'bibtex {}'.format(fname)
    bib_args = shlex.split(bib_cmd)
    p = subprocess.Popen(bib_args)

    p = subprocess.Popen(latex_args)
    p = subprocess.Popen(latex_args)

    p = subprocess.Popen(bib_args)

    clean_cmd = 'rm -f {0}.aux {0}.blg {0}.log {0}.dvi'.format(fname)
    clean_args = shlex.split(clean_cmd)
    print clean_args
    p = subprocess.Popen(clean_args)


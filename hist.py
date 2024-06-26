T-T,T-T-T,T-T-R,T-R-R,R-R-R,R-R-R,R-R-OW1,R-OW1-OW1,OW1-OW1-OW1,OW1-OW1-OW1,OW1-OW1-OW1,OW1-OW1-OW1,OW1-OW1-M,OW1-M-M,M-M-M,M-M-M,M-M-AH0,M-AH0-AH0,AH0-AH0-AH0,AH0-AH0-N,AH0-N-N,N-N-N,N-N-N,N-N-N,N-N-N,N-N-N,N-N-sil,N-sil-sil,sil-sil-sil,sil-sil-</s>'

In [29]:


# groupby
# dedup with groupby and record the count



In [30]: print(len(data))
28539

In [31]: with open('out.tsv', 'w') as fout:
    ...:     for idv, dat in tqdm(data, total=len(data)):
    ...:         print(idv + '\t' + make_seq_triphone(dat), file=fout)
    ...:
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28539/28539 [00:18<00:00, 1568.58it/s]

In [32]:
Do you really want to exit ([y]/n)? n

In [32]: hist
wi open(r'C:\Users\ChienChengChen\Desktop\mymeasure\phn_tsvs\train-clean-100.tsv') as f:
    data = f.read().strip().split('\n')
with open(r'C:\Users\ChienChengChen\Desktop\mymeasure\phn_tsvs\train-clean-100.tsv') as f:
    data = f.read().strip().split('\n')
data = [l.split('\t') for l in data]
from tqdm import tqdm
data[0]
data[0][0]
data[0][1]
data[0][2]
seq = data[0][1]
seq = seq.split(',')
print(seq[:20])
print((-1, 0, 1))
print((-1, 0, 1))
for i in range(10):
    print((i-1,i,i+1))
print((10-2,10-1,10))
print(([-1], 0, 1))
for i in range(1, 10-1):
    print((i-1,i,i+1))
print((10-2,10-1,[10]))



def make_seq_triphone(seq):
    triphones = []
    for i in range(len(seq)-2):
        triphones.append(seq[i:i+3])
    return triphones

make_seq_triphone("hello")



def make_seq_triphone(seq):
    triphones = [
        ('<s>', '<s>', seq[0]),
        ('<s>', seq[0], seq[1]),
    ]

    for i in range(len(seq)-2):
        triphones.append(seq[i:i+3])
    triphones.append(seq[-2], seq[-1], '</s>')
    triphones.append(seq[-1], '</s>', '</s>')
    return triphones

make_seq_triphone("hello")



def make_seq_triphone(seq):
    triphones = [
        ('<s>', '<s>', seq[0]),
        ('<s>', seq[0], seq[1]),
    ]

    for i in range(len(seq)-2):
        triphones.append(seq[i:i+3])
    triphones.append((seq[-2], seq[-1], '</s>'))
    triphones.append((seq[-1], '</s>', '</s>'))
    return triphones

make_seq_triphone("hello")


def triphonize(s):
    return "-".join(s)

def make_seq_triphone(seq):
    triphones = [
        triphonize('<s>', seq[0], seq[1]),
    ]

    for i in range(len(seq)-2):
        triphones.append(triphonize(seq[i:i+3]))
    triphones.append(triphonize(seq[-2], seq[-1], '</s>'))
    return triphones

make_seq_triphone("hello")
seq
data[0][1]


def triphonize(a, b, c):
    return f"{a}-{b}-{c}"

def make_seq_triphone(seqin):
    seq = seqin.split(',')
    triphones = [
        triphonize('<s>', seq[0], seq[1]),
    ]

    for i in range(len(seq)-2):
        triphones.append(triphonize(seq[i:i+3]))
    triphones.append(triphonize(seq[-2], seq[-1], '</s>'))
    return ','.join(triphones)

make_seq_triphone('sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,DH,DH,DH,EH1,EH1,R,R,W,W,W,AA1,AA1,AA1,AA1,AA1,AA1,AA1,AA1,AA1,Z,Z,Z,Z,Z,AH1,AH1,V,V,V,V,V,K,K,K,K,K,AO1,AO1,AO1,R,R,R,R,R,R,S,S,S,S,N,N,N,N,OW1,OW1,OW1,OW1,L,L,L,L,L,IY0,IY0,IY0,IY0,G,G,G,G,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,L,L,L,AH0,AH0,AH0,T,T,IY0,IY0,IY0,IY0,IY0,IH0,IH0,IH0,N,N,DH,DH,DH,IY0,IY0,IY0,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,K,K,K,T,T,T,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,N,N,D,D,D,K,K,K,K,AA1,AA1,AA1,R,R,R,R,R,L,L,L,L,DH,DH,AH0,AH0,AH0,G,G,G,R,R,R,R,EY1,EY1,EY1,EY1,EY1,EY1,EY1,EY1,EY1,EY1,T,T,T,sil,W,W,W,W,AH1,AH1,AH1,AH1,AH1,Z,Z,Z,Z,IH1,IH1,IH1,IH1,N,N,N,N,N,N,N,OW1,OW1,OW1,OW1,OW1,OW1,OW1,R,R,R,R,IY1,IY1,IY1,IY1,L,L,L,L,L,S,S,S,S,S,S,EH1,EH1,EH1,EH1,EH1,N,N,N,N,S,S,S,S,DH,DH,DH,AH0,AH0,AH0,AH0,S,S,S,S,S,AH0,AH0,AH0,K,K,K,S,S,S,S,EH1,EH1,EH1,EH1,EH1,EH1,EH1,S,S,S,S,S,S,ER0,ER0,ER0,ER0,ER0,AH0,AH0,AH0,V,V,AA0,AA0,AA0,AA0,AA0,AA0,N,N,N,N,ER1,ER1,ER1,ER1,ER1,ER1,ER1,ER1,ER1,ER1,ER1,IY0,IY0,IY0,IY0,IY0,IY0,IH0,IH0,IH0,IH0,IH0,IH0,S,S,S,S,S,S,S,S,sil,AE1,AE1,AE1,AE1,N,N,N,D,D,R,R,R,AA1,AA1,AA1,AA1,AA1,AA1,M,M,M,Y,Y,Y,AH0,AH0,L,L,L,L,AH0,AH0,AH0,AH0,S,S,S,AH0,AH0,AH0,AH0,G,G,G,AH1,AH1,AH1,AH1,AH1,AH1,S,S,S,T,T,T,T,T,AH0,AH0,AH0,AH0,L,L,L,L,L,AH0,AH0,AH0,AH0,AH0,S,S,S,S,S,S,S,S,S,S,sil,sil,sil,sil,sil,B,B,B,AH1,AH1,T,T,HH,IY1,IY1,IY1,IY1,IY1,R,R,R,UW1,UW1,UW1,UW1,UW1,UW1,UW1,UW1,L,L,L,D,D,D,AH0,AH0,AH0,AH0,G,G,G,R,R,UW1,UW1,UW1,UW1,P,P,P,AH0,V,V,V,K,K,K,K,K,K,IH1,IH1,IH1,NG,NG,NG,NG,NG,D,D,AH0,AH0,AH0,M,M,M,Z,Z,Z,Z,W,W,W,IH1,IH1,IH1,CH,CH,CH,CH,EH0,EH0,EH0,M,M,M,B,B,B,R,R,R,R,EY1,EY1,EY1,EY1,EY1,EY1,EY1,S,S,S,S,T,T,DH,DH,AH0,AH0,L,L,L,L,L,L,AA1,AA1,AA1,AA1,AA1,AA1,R,R,R,R,R,JH,JH,JH,JH,JH,JH,ER0,ER0,ER0,HH,HH,HH,HH,HH,HH,HH,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,AE1,F,F,F,F,F,AH0,V,V,V,V,DH,DH,IY0,IY0,IY0,IY0,IY0,OW1,OW1,OW1,OW1,OW1,OW1,L,L,L,L,L,L,L,L,L,D,D,D,D,W,W,W,W,W,EH1,EH1,EH1,EH1,S,S,S,T,T,T,T,T,ER0,ER0,N,N,N,N,EH1,EH1,EH1,EH1,M,M,M,M,P,P,P,P,P,AY0,AY0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,sil,sil,sil,AH0,AH0,AH0,N,N,D,D,D,D,F,F,F,AO1,AO1,AO1,AO1,R,R,R,M,M,D,D,D,AH0,AH0,AH0,F,F,F,F,F,F,EH1,EH1,EH1,EH1,EH1,EH1,EH1,EH1,EH1,EH1,R,R,R,IH0,IH0,IH0,IH0,IH0,K,K,K,K,K,W,W,IH1,IH1,P,P,P,P,P,P,P,P,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,OY1,Z,Z,Z,T,T,T,AH0,AH0,AH0,DH,DH,AH0,AH0,AH0,AH0,R,R,R,R,EH1,EH1,EH1,EH1,EH1,L,L,L,L,L,M,M,M,M,M,M,M,N,N,N,N,AW1,AW1,AW1,AW1,AW1,AW1,AW1,AW1,AW1,AW1,R,R,R,R,R,R,R,UW1,UW1,UW1,UW1,UW1,L,L,L,L,L,L,L,D,D,D,B,B,B,B,AY1,AY1,AY1,AY1,AY1,AY1,AY1,AY1,AY1,AY0,AY0,AY0,AY0,AY0,AY0,R,R,R,R,R,R,R,IY1,IY1,IY1,IY1,IY1,IY1,IY1,IY1,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,sil,F,F,F,F,R,AH1,AH1,M,M,M,EY1,EY1,EY1,EY1,EY1,EY1,T,T,HH,HH,HH,HH,AH1,AH1,AH1,N,D,D,R,R,IH0,IH0,IH0,D,D,D,DH,DH,EH1,EH1,EH1,EH1,EH1,EH1,EH1,N,N,N,AO1,AO1,AO1,AO1,AO1,AO1,AO1,AO1,AO1,N,N,N,N,W,W,W,W,W,ER0,ER0,ER0,ER0,ER0,ER0,D,D,D,W,W,W,IY1,IY1,IY1,IY1,IY1,HH,HH,HH,HH,HH,AE1,AE1,AE1,AE1,AE1,AE1,V,V,W,W,W,W,W,W,AH1,AH1,AH1,AH1,N,N,N,S,S,S,S,M,M,M,M,M,M,AO1,AO1,AO1,AO1,AO1,R,R,R,R,AH0,AH0,AH0,AH0,AH0,W,W,W,W,W,W,EH1,EH1,EH1,EH1,EH1,EH1,EH1,S,S,S,S,S,T,T,T,T,T,T,T,T,R,R,R,R,OW1,OW1,OW1,OW1,OW1,M,M,M,AH0,AH0,N,N,N,N,EH1,EH1,EH1,EH1,EH1,M,M,M,M,P,P,P,P,P,P,AY0,AY0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,ER0,IH1,IH1,IH1,N,N,N,IH0,IH0,IH0,G,G,G,G,Z,Z,Z,IH1,IH1,IH1,IH1,S,S,S,T,T,T,T,AH0,AH0,N,N,N,N,N,N,N,N,N,S,S,S,S,S,S,S,AE1,AE1,AE1,AE1,AE1,Z,Z,Z,Z,W,W,W,W,EH1,EH1,EH1,EH1,EH1,EH1,EH1,L,L,AE1,AE1,AE1,Z,Z,Z,DH,DH,DH,IY0,IY0,IY0,IY1,IY1,IY1,IY1,IY1,IY1,IY1,IY1,S,S,S,T,T,T,T,R,R,R,R,OW1,OW1,OW1,OW1,OW1,OW1,M,M,M,M,AH0,AH0,AH0,N,N,N,N,N,N,N,sil,sil,sil')
%edit
AA


def triphonize(a, b, c):
    return f"{a}-{b}-{c}"

def make_seq_triphone(seqin):
    seq = seqin.split(',')
    triphones = [
        triphonize('<s>', seq[0], seq[1]),
    ]

    for i in range(len(seq)-2):
        triphones.append(triphonize(*seq[i:i+3]))
    triphones.append(triphonize(seq[-2], seq[-1], '</s>'))
    return ','.join(triphones)

make_seq_triphone(AA)
len(_24.split(','))
AA.split(',')
len(AA.split(','))
_24

def triphonize(a, b, c):
    return f"{a}-{b}-{c}"

def make_seq_triphone(seqin):
    seq = seqin.split(',')
    triphones = [
        triphonize('sta', seq[0], seq[1]),
    ]

    for i in range(len(seq)-2):
        triphones.append(triphonize(*seq[i:i+3]))
    triphones.append(triphonize(seq[-2], seq[-1], 'stp'))
    return ','.join(triphones)
print(len(data))
with open('out.tsv', 'w') as fout:
    for idv, dat in tqdm(data, total=len(data)):
        print(idv + '\t' + make_seq_triphone(dat), file=fout)
hist


20:52:56 mymeasure > pyth^C
20:52:58 mymeasure > foreach ($mod in @("hubert", "w2v2")) { foreach ($el in @("050", "100", "200")) { python mymeasure.py   flists   unit_hyps/$mod/clu$el   unit   --phn_dir trip_tsvs   --lab_sets train-clean-100   --phn_sets train-clean-100   --verbose --output tri_${mod}_$el.npz } }
Traceback (most recent call last):
  File "C:\Users\ChienChengChen\Desktop\mymeasure\mymeasure.py", line 390, in <module>    
    main_phn_lab(
  File "C:\Users\ChienChengChen\Desktop\mymeasure\mymeasure.py", line 336, in main_phn_lab
    uid2refs.update(read_phn(f"{phn_dir}/{s}.tsv"))
  File "C:\Users\ChienChengChen\Desktop\mymeasure\mymeasure.py", line 272, in read_phn    
    phns = [re.sub("[0-9]", "", phn) for phn in phns]
  File "C:\Users\ChienChengChen\Desktop\mymeasure\mymeasure.py", line 272, in <listcomp>  
    phns = [re.sub("[0-9]", "", phn) for phn in phns]
  File "C:\Users\JeffChen\miniconda3\lib\re.py", line 209, in sub
    return _compile(pattern, flags).sub(repl, string, count)
KeyboardInterrupt
20:54:32 mymeasure > ipython3
Python 3.10.12 | packaged by Anaconda, Inc. | (main, Jul  5 2023, 19:01:18) [MSC v.1916 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.15.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: with open('c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.ts 
   ...: v') as f: data = f.read()
  Cell In[1], line 1
    with open('c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.tsv') as f: data = f.read()
                                                                                       ^  
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape


In [2]: with open(r'c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.t 
   ...: sv') as f: data = f.read()

In [3]: data = data.split('\n')

In [4]: data = [l.split('\t') for l in data]

In [5]: import re
   ...: data = [(a,re.sub(r'\d', '', b))for a, b in data]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[5], line 2
      1 import re
----> 2 data = [(a,re.sub(r'\d', '', b))for a, b in data]

Cell In[5], line 2, in <listcomp>(.0)
      1 import re
----> 2 data = [(a,re.sub(r'\d', '', b))for a, b in data]

ValueError: not enough values to unpack (expected 2, got 1)

In [6]: rep 2 3 4 5

In [7]: with open(r'c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.t 
   ...: sv') as f: data = f.read()
   ...: data = data.split('\n')
   ...: data = [l.split('\t') for l in data]
   ...: import re; from tqdm import tqdm
   ...: data = [(a[0],re.sub(r'\d', '', a[1])) if len(a) == 2 else a for a in tqdm(data)] 
   ...:
100%|████████████████████████████████████████████| 28540/28540 [00:05<00:00, 5541.07it/s]

In [8]: z = '\n'.join(['\t'.join(l) for l in data])

In [9]: with open(r'c:/Users/ChienChengChen/Desktop/mymeasure/trip_tsvs/train-clean-100.t 
   ...: sv
  Cell In[9], line 1
    with open(r'c:/Users/ChienChengChen/Desktop/mymeasure/trip_tsvs/train-clean-100.tsv   
              ^
SyntaxError: unterminated string literal (detected at line 1)


In [10]: with open(r'c:/Users/ChienChengChen/Desktop/mymeasure/trip_tsvs/train-clean-100. 
    ...: tsv', 'w') as f:
    ...:     f.write(z)
    ...:

In [11]: hist
with open('c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.tsv') as f: data = f.read()
with open(r'c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.tsv') as f: data = f.read()
data = data.split('\n')
data = [l.split('\t') for l in data]
import re
data = [(a,re.sub(r'\d', '', b))for a, b in data]
rep 2 3 4 5
with open(r'c:\Users\ChienChengChen\Desktop\mymeasure\trip_tsvs\train-clean-100.tsv') as f: data = f.read()
data = data.split('\n')
data = [l.split('\t') for l in data]
import re; from tqdm import tqdm
data = [(a[0],re.sub(r'\d', '', a[1])) if len(a) == 2 else a for a in tqdm(data)]
z = '\n'.join(['\t'.join(l) for l in data])
with open(r'c:/Users/ChienChengChen/Desktop/mymeasure/trip_tsvs/train-clean-100.tsv       
with open(r'c:/Users/ChienChengChen/Desktop/mymeasure/trip_tsvs/train-clean-100.tsv', 'w') as f:
    f.write(z)
hist

######################
 ('sil', 'sta', 'NG'),
 ('sil', 'sta', 'L'),
 ('sil', 'sta', 'R'),
 ('sil', 'sta', 'Y'),
 ('sil', 'sta', 'W'),
 ('sil', 'sta', 'IY'),
 ('sil', 'sta', 'IH'),
 ('sil', 'sta', 'UW'),
 ('sil', 'sta', 'UH'),
 ('sil', 'sta', 'EH'),
 ('sil', 'sta', 'ER'),
 ('sil', 'sta', 'AO'),
 ('sil', 'sta', 'AE'),
 ('sil', 'sta', 'AH'),
 ('sil', 'sta', 'AA'),
 ('sil', 'sta', 'EY'),
 ('sil', 'sta', 'OW'),
 ('sil', 'sta', 'OY'),
 ('sil', 'sta', 'AY'),
 ('sil', 'sta', 'AW'),
 ('sil', 'sta', 'stp'),
 ('spn', 'sta', 'sta'),
 ('spn', 'sta', 'sil'),
 ('spn', 'sta', 'spn'),
 ('spn', 'sta', 'P'),
 ('spn', 'sta', 'T'),
 ('spn', 'sta', 'K'),
 ('spn', 'sta', 'B'),
 ('spn', 'sta', 'D'),
 ('spn', 'sta', 'G'),
 ('spn', 'sta', 'CH'),
 ('spn', 'sta', 'JH'),
 ('spn', 'sta', 'F'),
 ('spn', 'sta', 'S'),
 ('spn', 'sta', 'TH'),
 ('spn', 'sta', 'SH'),
 ('spn', 'sta', 'HH'),
 ('spn', 'sta', 'V'),
 ('spn', 'sta', 'Z'),
 ('spn', 'sta', 'DH'),
 ('spn', 'sta', 'ZH'),
 ('spn', 'sta', 'M'),
 ('spn', 'sta', 'N'),
 ('spn', 'sta', 'NG'),
 ('spn', 'sta', 'L'),
 ('spn', 'sta', 'R'),
 ('spn', 'sta', 'Y'),
 ('spn', 'sta', 'W'),
 ('spn', 'sta', 'IY'),
 ('spn', 'sta', 'IH'),
 ('spn', 'sta', 'UW'),
 ('spn', 'sta', 'UH'),
 ('spn', 'sta', 'EH'),
 ('spn', 'sta', 'ER'),
 ('spn', 'sta', 'AO'),
 ('spn', 'sta', 'AE'),
 ('spn', 'sta', 'AH'),
 ('spn', 'sta', 'AA'),
 ('spn', 'sta', 'EY'),
 ('spn', 'sta', 'OW'),
 ('spn', 'sta', 'OY'),
 ('spn', 'sta', 'AY'),
 ('spn', 'sta', 'AW'),
 ('spn', 'sta', 'stp'),
 ('P', 'sta', 'sta'),
 ('P', 'sta', 'sil'),
 ('P', 'sta', 'spn'),
 ('P', 'sta', 'P'),
 ('P', 'sta', 'T'),
 ('P', 'sta', 'K'),
 ('P', 'sta', 'B'),
 ('P', 'sta', 'D'),
 ('P', 'sta', 'G'),
 ('P', 'sta', 'CH'),
 ('P', 'sta', 'JH'),
 ('P', 'sta', 'F'),
 ('P', 'sta', 'S'),
 ('P', 'sta', 'TH'),
 ('P', 'sta', 'SH'),
 ('P', 'sta', 'HH'),
 ('P', 'sta', 'V'),
 ('P', 'sta', 'Z'),
 ('P', 'sta', 'DH'),
 ('P', 'sta', 'ZH'),
 ('P', 'sta', 'M'),
 ('P', 'sta', 'N'),
 ('P', 'sta', 'NG'),
 ('P', 'sta', 'L'),
 ('P', 'sta', 'R'),
 ('P', 'sta', 'Y'),
 ('P', 'sta', 'W'),
 ('P', 'sta', 'IY'),
 ('P', 'sta', 'IH'),
 ('P', 'sta', 'UW'),
 ('P', 'sta', 'UH'),
 ('P', 'sta', 'EH'),
 ('P', 'sta', 'ER'),
 ('P', 'sta', 'AO'),
 ('P', 'sta', 'AE'),
 ('P', 'sta', 'AH'),
 ('P', 'sta', 'AA'),
 ('P', 'sta', 'EY'),
 ('P', 'sta', 'OW'),
 ('P', 'sta', 'OY'),
 ('P', 'sta', 'AY'),
 ('P', 'sta', 'AW'),
 ('P', 'sta', 'stp'),
 ('T', 'sta', 'sta'),
 ('T', 'sta', 'sil'),
 ('T', 'sta', 'spn'),
 ('T', 'sta', 'P'),
 ('T', 'sta', 'T'),
 ('T', 'sta', 'K'),
 ('T', 'sta', 'B'),
 ('T', 'sta', 'D'),
 ('T', 'sta', 'G'),
 ('T', 'sta', 'CH'),
 ('T', 'sta', 'JH'),
 ('T', 'sta', 'F'),
 ('T', 'sta', 'S'),
 ('T', 'sta', 'TH'),
 ('T', 'sta', 'SH'),
 ('T', 'sta', 'HH'),
 ('T', 'sta', 'V'),
 ('T', 'sta', 'Z'),
 ('T', 'sta', 'DH'),
 ('T', 'sta', 'ZH'),
 ('T', 'sta', 'M'),
 ('T', 'sta', 'N'),
 ('T', 'sta', 'NG'),
 ('T', 'sta', 'L'),
 ('T', 'sta', 'R'),
 ('T', 'sta', 'Y'),
 ('T', 'sta', 'W'),
 ('T', 'sta', 'IY'),
 ('T', 'sta', 'IH'),
 ('T', 'sta', 'UW'),
 ('T', 'sta', 'UH'),
 ('T', 'sta', 'EH'),
 ('T', 'sta', 'ER'),
 ('T', 'sta', 'AO'),
 ('T', 'sta', 'AE'),
 ('T', 'sta', 'AH'),
 ('T', 'sta', 'AA'),
 ('T', 'sta', 'EY'),
 ('T', 'sta', 'OW'),
 ('T', 'sta', 'OY'),
 ('T', 'sta', 'AY'),
 ('T', 'sta', 'AW'),
 ('T', 'sta', 'stp'),
 ('K', 'sta', 'sta'),
 ('K', 'sta', 'sil'),
 ('K', 'sta', 'spn'),
 ('K', 'sta', 'P'),
 ('K', 'sta', 'T'),
 ('K', 'sta', 'K'),
 ('K', 'sta', 'B'),
 ('K', 'sta', 'D'),
 ('K', 'sta', 'G'),
 ('K', 'sta', 'CH'),
 ('K', 'sta', 'JH'),
 ('K', 'sta', 'F'),
 ('K', 'sta', 'S'),
 ('K', 'sta', 'TH'),
 ('K', 'sta', 'SH'),
 ('K', 'sta', 'HH'),
 ('K', 'sta', 'V'),
 ('K', 'sta', 'Z'),
 ('K', 'sta', 'DH'),
 ('K', 'sta', 'ZH'),
 ('K', 'sta', 'M'),
 ('K', 'sta', 'N'),
 ('K', 'sta', 'NG'),
 ('K', 'sta', 'L'),
 ('K', 'sta', 'R'),
 ('K', 'sta', 'Y'),
 ('K', 'sta', 'W'),
 ('K', 'sta', 'IY'),
 ('K', 'sta', 'IH'),
 ('K', 'sta', 'UW'),
 ('K', 'sta', 'UH'),
 ('K', 'sta', 'EH'),
 ('K', 'sta', 'ER'),
 ('K', 'sta', 'AO'),
 ('K', 'sta', 'AE'),
 ('K', 'sta', 'AH'),
 ('K', 'sta', 'AA'),
 ('K', 'sta', 'EY'),
 ('K', 'sta', 'OW'),
 ('K', 'sta', 'OY'),
 ('K', 'sta', 'AY'),
 ('K', 'sta', 'AW'),
 ('K', 'sta', 'stp'),
 ('B', 'sta', 'sta'),
 ('B', 'sta', 'sil'),
 ('B', 'sta', 'spn'),
 ('B', 'sta', 'P'),
 ('B', 'sta', 'T'),
 ('B', 'sta', 'K'),
 ('B', 'sta', 'B'),
 ('B', 'sta', 'D'),
 ('B', 'sta', 'G'),
 ('B', 'sta', 'CH'),
 ('B', 'sta', 'JH'),
 ('B', 'sta', 'F'),
 ('B', 'sta', 'S'),
 ('B', 'sta', 'TH'),
 ('B', 'sta', 'SH'),
 ('B', 'sta', 'HH'),
 ('B', 'sta', 'V'),
 ('B', 'sta', 'Z'),
 ('B', 'sta', 'DH'),
 ('B', 'sta', 'ZH'),
 ('B', 'sta', 'M'),
 ('B', 'sta', 'N'),
 ('B', 'sta', 'NG'),
 ('B', 'sta', 'L'),
 ('B', 'sta', 'R'),
 ('B', 'sta', 'Y'),
 ('B', 'sta', 'W'),
 ('B', 'sta', 'IY'),
 ('B', 'sta', 'IH'),
 ('B', 'sta', 'UW'),
 ('B', 'sta', 'UH'),
 ('B', 'sta', 'EH'),
 ('B', 'sta', 'ER'),
 ('B', 'sta', 'AO'),
 ('B', 'sta', 'AE'),
 ('B', 'sta', 'AH'),
 ('B', 'sta', 'AA'),
 ('B', 'sta', 'EY'),
 ('B', 'sta', 'OW'),
 ('B', 'sta', 'OY'),
 ('B', 'sta', 'AY'),
 ('B', 'sta', 'AW'),
 ('B', 'sta', 'stp'),
 ('D', 'sta', 'sta'),
 ('D', 'sta', 'sil'),
 ('D', 'sta', 'spn'),
 ('D', 'sta', 'P'),
 ('D', 'sta', 'T'),
 ('D', 'sta', 'K'),
 ('D', 'sta', 'B'),
 ('D', 'sta', 'D'),
 ('D', 'sta', 'G'),
 ('D', 'sta', 'CH'),
 ('D', 'sta', 'JH'),
 ('D', 'sta', 'F'),
 ('D', 'sta', 'S'),
 ('D', 'sta', 'TH'),
 ('D', 'sta', 'SH'),
 ('D', 'sta', 'HH'),
 ('D', 'sta', 'V'),
 ('D', 'sta', 'Z'),
 ('D', 'sta', 'DH'),
 ('D', 'sta', 'ZH'),
 ('D', 'sta', 'M'),
 ('D', 'sta', 'N'),
 ('D', 'sta', 'NG'),
 ('D', 'sta', 'L'),
 ('D', 'sta', 'R'),
 ('D', 'sta', 'Y'),
 ('D', 'sta', 'W'),
 ('D', 'sta', 'IY'),
 ('D', 'sta', 'IH'),
 ('D', 'sta', 'UW'),
 ('D', 'sta', 'UH'),
 ('D', 'sta', 'EH'),
 ('D', 'sta', 'ER'),
 ('D', 'sta', 'AO'),
 ('D', 'sta', 'AE'),
 ('D', 'sta', 'AH'),
 ('D', 'sta', 'AA'),
 ('D', 'sta', 'EY'),
 ('D', 'sta', 'OW'),
 ('D', 'sta', 'OY'),
 ('D', 'sta', 'AY'),
 ('D', 'sta', 'AW'),
 ('D', 'sta', 'stp'),
 ('G', 'sta', 'sta'),
 ('G', 'sta', 'sil'),
 ('G', 'sta', 'spn'),
 ('G', 'sta', 'P'),
 ('G', 'sta', 'T'),
 ('G', 'sta', 'K'),
 ('G', 'sta', 'B'),
 ('G', 'sta', 'D'),
 ('G', 'sta', 'G'),
 ('G', 'sta', 'CH'),
 ('G', 'sta', 'JH'),
 ('G', 'sta', 'F'),
 ('G', 'sta', 'S'),
 ('G', 'sta', 'TH'),
 ('G', 'sta', 'SH'),
 ('G', 'sta', 'HH'),
 ('G', 'sta', 'V'),
 ('G', 'sta', 'Z'),
 ('G', 'sta', 'DH'),
 ('G', 'sta', 'ZH'),
 ('G', 'sta', 'M'),
 ('G', 'sta', 'N'),
 ('G', 'sta', 'NG'),
 ('G', 'sta', 'L'),
 ('G', 'sta', 'R'),
 ('G', 'sta', 'Y'),
 ('G', 'sta', 'W'),
 ('G', 'sta', 'IY'),
 ('G', 'sta', 'IH'),
 ('G', 'sta', 'UW'),
 ('G', 'sta', 'UH'),
 ('G', 'sta', 'EH'),
 ('G', 'sta', 'ER'),
 ('G', 'sta', 'AO'),
 ('G', 'sta', 'AE'),
 ('G', 'sta', 'AH'),
 ('G', 'sta', 'AA'),
 ('G', 'sta', 'EY'),
 ('G', 'sta', 'OW'),
 ('G', 'sta', 'OY'),
 ('G', 'sta', 'AY'),
 ('G', 'sta', 'AW'),
 ('G', 'sta', 'stp'),
 ('CH', 'sta', 'sta'),
 ('CH', 'sta', 'sil'),
 ('CH', 'sta', 'spn'),
 ('CH', 'sta', 'P'),
 ('CH', 'sta', 'T'),
 ('CH', 'sta', 'K'),
 ('CH', 'sta', 'B'),
 ('CH', 'sta', 'D'),
 ('CH', 'sta', 'G'),
 ('CH', 'sta', 'CH'),
 ('CH', 'sta', 'JH'),
 ('CH', 'sta', 'F'),
 ('CH', 'sta', 'S'),
 ('CH', 'sta', 'TH'),
 ('CH', 'sta', 'SH'),
 ('CH', 'sta', 'HH'),
 ('CH', 'sta', 'V'),
 ('CH', 'sta', 'Z'),
 ('CH', 'sta', 'DH'),
 ('CH', 'sta', 'ZH'),
 ('CH', 'sta', 'M'),
 ('CH', 'sta', 'N'),
 ('CH', 'sta', 'NG'),
 ('CH', 'sta', 'L'),
 ('CH', 'sta', 'R'),
 ('CH', 'sta', 'Y'),
 ('CH', 'sta', 'W'),
 ('CH', 'sta', 'IY'),
 ('CH', 'sta', 'IH'),
 ('CH', 'sta', 'UW'),
 ('CH', 'sta', 'UH'),
 ('CH', 'sta', 'EH'),
 ('CH', 'sta', 'ER'),
 ('CH', 'sta', 'AO'),
 ('CH', 'sta', 'AE'),
 ('CH', 'sta', 'AH'),
 ('CH', 'sta', 'AA'),
 ('CH', 'sta', 'EY'),
 ('CH', 'sta', 'OW'),
 ('CH', 'sta', 'OY'),
 ('CH', 'sta', 'AY'),
 ('CH', 'sta', 'AW'),
 ('CH', 'sta', 'stp'),
 ('JH', 'sta', 'sta'),
 ('JH', 'sta', 'sil'),
 ('JH', 'sta', 'spn'),
 ('JH', 'sta', 'P'),
 ('JH', 'sta', 'T'),
 ('JH', 'sta', 'K'),
 ('JH', 'sta', 'B'),
 ('JH', 'sta', 'D'),
 ('JH', 'sta', 'G'),
 ('JH', 'sta', 'CH'),
 ('JH', 'sta', 'JH'),
 ('JH', 'sta', 'F'),
 ('JH', 'sta', 'S'),
 ('JH', 'sta', 'TH'),
 ('JH', 'sta', 'SH'),
 ('JH', 'sta', 'HH'),
 ('JH', 'sta', 'V'),
 ('JH', 'sta', 'Z'),
 ('JH', 'sta', 'DH'),
 ('JH', 'sta', 'ZH'),
 ('JH', 'sta', 'M'),
 ('JH', 'sta', 'N'),
 ('JH', 'sta', 'NG'),
 ('JH', 'sta', 'L'),
 ('JH', 'sta', 'R'),
 ('JH', 'sta', 'Y'),
 ('JH', 'sta', 'W'),
 ('JH', 'sta', 'IY'),
 ('JH', 'sta', 'IH'),
 ('JH', 'sta', 'UW'),
 ('JH', 'sta', 'UH'),
 ('JH', 'sta', 'EH'),
 ('JH', 'sta', 'ER'),
 ('JH', 'sta', 'AO'),
 ('JH', 'sta', 'AE'),
 ('JH', 'sta', 'AH'),
 ('JH', 'sta', 'AA'),
 ('JH', 'sta', 'EY'),
 ('JH', 'sta', 'OW'),
 ('JH', 'sta', 'OY'),
 ('JH', 'sta', 'AY'),
 ('JH', 'sta', 'AW'),
 ('JH', 'sta', 'stp'),
 ('F', 'sta', 'sta'),
 ('F', 'sta', 'sil'),
 ('F', 'sta', 'spn'),
 ('F', 'sta', 'P'),
 ('F', 'sta', 'T'),
 ('F', 'sta', 'K'),
 ('F', 'sta', 'B'),
 ('F', 'sta', 'D'),
 ('F', 'sta', 'G'),
 ('F', 'sta', 'CH'),
 ('F', 'sta', 'JH'),
 ('F', 'sta', 'F'),
 ('F', 'sta', 'S'),
 ('F', 'sta', 'TH'),
 ('F', 'sta', 'SH'),
 ('F', 'sta', 'HH'),
 ('F', 'sta', 'V'),
 ('F', 'sta', 'Z'),
 ('F', 'sta', 'DH'),
 ('F', 'sta', 'ZH'),
 ('F', 'sta', 'M'),
 ('F', 'sta', 'N'),
 ('F', 'sta', 'NG'),
 ('F', 'sta', 'L'),
 ('F', 'sta', 'R'),
 ('F', 'sta', 'Y'),
 ('F', 'sta', 'W'),
 ('F', 'sta', 'IY'),
 ('F', 'sta', 'IH'),
 ('F', 'sta', 'UW'),
 ('F', 'sta', 'UH'),
 ('F', 'sta', 'EH'),
 ('F', 'sta', 'ER'),
 ('F', 'sta', 'AO'),
 ('F', 'sta', 'AE'),
 ('F', 'sta', 'AH'),
 ('F', 'sta', 'AA'),
 ('F', 'sta', 'EY'),
 ('F', 'sta', 'OW'),
 ('F', 'sta', 'OY'),
 ('F', 'sta', 'AY'),
 ('F', 'sta', 'AW'),
 ('F', 'sta', 'stp'),
 ('S', 'sta', 'sta'),
 ('S', 'sta', 'sil'),
 ('S', 'sta', 'spn'),
 ('S', 'sta', 'P'),
 ('S', 'sta', 'T'),
 ('S', 'sta', 'K'),
 ('S', 'sta', 'B'),
 ('S', 'sta', 'D'),
 ('S', 'sta', 'G'),
 ('S', 'sta', 'CH'),
 ('S', 'sta', 'JH'),
 ('S', 'sta', 'F'),
 ('S', 'sta', 'S'),
 ('S', 'sta', 'TH'),
 ('S', 'sta', 'SH'),
 ('S', 'sta', 'HH'),
 ('S', 'sta', 'V'),
 ('S', 'sta', 'Z'),
 ('S', 'sta', 'DH'),
 ('S', 'sta', 'ZH'),
 ('S', 'sta', 'M'),
 ('S', 'sta', 'N'),
 ('S', 'sta', 'NG'),
 ('S', 'sta', 'L'),
 ('S', 'sta', 'R'),
 ('S', 'sta', 'Y'),
 ('S', 'sta', 'W'),
 ('S', 'sta', 'IY'),
 ('S', 'sta', 'IH'),
 ('S', 'sta', 'UW'),
 ('S', 'sta', 'UH'),
 ('S', 'sta', 'EH'),
 ('S', 'sta', 'ER'),
 ('S', 'sta', 'AO'),
 ('S', 'sta', 'AE'),
 ('S', 'sta', 'AH'),
 ('S', 'sta', 'AA'),
 ('S', 'sta', 'EY'),
 ('S', 'sta', 'OW'),
 ('S', 'sta', 'OY'),
 ('S', 'sta', 'AY'),
 ('S', 'sta', 'AW'),
 ('S', 'sta', 'stp'),
 ('TH', 'sta', 'sta'),
 ('TH', 'sta', 'sil'),
 ('TH', 'sta', 'spn'),
 ('TH', 'sta', 'P'),
 ('TH', 'sta', 'T'),
 ('TH', 'sta', 'K'),
 ('TH', 'sta', 'B'),
 ('TH', 'sta', 'D'),
 ('TH', 'sta', 'G'),
 ('TH', 'sta', 'CH'),
 ('TH', 'sta', 'JH'),
 ('TH', 'sta', 'F'),
 ('TH', 'sta', 'S'),
 ('TH', 'sta', 'TH'),
 ('TH', 'sta', 'SH'),
 ('TH', 'sta', 'HH'),
 ('TH', 'sta', 'V'),
 ('TH', 'sta', 'Z'),
 ('TH', 'sta', 'DH'),
 ('TH', 'sta', 'ZH'),
 ('TH', 'sta', 'M'),
 ('TH', 'sta', 'N'),
 ('TH', 'sta', 'NG'),
 ('TH', 'sta', 'L'),
 ('TH', 'sta', 'R'),
 ('TH', 'sta', 'Y'),
 ('TH', 'sta', 'W'),
 ('TH', 'sta', 'IY'),
 ('TH', 'sta', 'IH'),
 ('TH', 'sta', 'UW'),
 ('TH', 'sta', 'UH'),
 ('TH', 'sta', 'EH'),
 ('TH', 'sta', 'ER'),
 ('TH', 'sta', 'AO'),
 ('TH', 'sta', 'AE'),
 ('TH', 'sta', 'AH'),
 ('TH', 'sta', 'AA'),
 ('TH', 'sta', 'EY'),
 ('TH', 'sta', 'OW'),
 ('TH', 'sta', 'OY'),
 ('TH', 'sta', 'AY'),
 ('TH', 'sta', 'AW'),
 ('TH', 'sta', 'stp'),
 ('SH', 'sta', 'sta'),
 ('SH', 'sta', 'sil'),
 ('SH', 'sta', 'spn'),
 ('SH', 'sta', 'P'),
 ('SH', 'sta', 'T'),
 ('SH', 'sta', 'K'),
 ('SH', 'sta', 'B'),
 ('SH', 'sta', 'D'),
 ('SH', 'sta', 'G'),
 ('SH', 'sta', 'CH'),
 ('SH', 'sta', 'JH'),
 ('SH', 'sta', 'F'),
 ('SH', 'sta', 'S'),
 ('SH', 'sta', 'TH'),
 ('SH', 'sta', 'SH'),
 ('SH', 'sta', 'HH'),
 ('SH', 'sta', 'V'),
 ('SH', 'sta', 'Z'),
 ('SH', 'sta', 'DH'),
 ('SH', 'sta', 'ZH'),
 ('SH', 'sta', 'M'),
 ('SH', 'sta', 'N'),
 ('SH', 'sta', 'NG'),
 ('SH', 'sta', 'L'),
 ('SH', 'sta', 'R'),
 ('SH', 'sta', 'Y'),
 ('SH', 'sta', 'W'),
 ('SH', 'sta', 'IY'),
 ('SH', 'sta', 'IH'),
 ('SH', 'sta', 'UW'),
 ('SH', 'sta', 'UH'),
 ('SH', 'sta', 'EH'),
 ('SH', 'sta', 'ER'),
 ('SH', 'sta', 'AO'),
 ('SH', 'sta', 'AE'),
 ('SH', 'sta', 'AH'),
 ('SH', 'sta', 'AA'),
 ('SH', 'sta', 'EY'),
 ('SH', 'sta', 'OW'),
 ('SH', 'sta', 'OY'),
 ('SH', 'sta', 'AY'),
 ('SH', 'sta', 'AW'),
 ('SH', 'sta', 'stp'),
 ('HH', 'sta', 'sta'),
 ('HH', 'sta', 'sil'),
 ('HH', 'sta', 'spn'),
 ('HH', 'sta', 'P'),
 ('HH', 'sta', 'T'),
 ('HH', 'sta', 'K'),
 ('HH', 'sta', 'B'),
 ('HH', 'sta', 'D'),
 ('HH', 'sta', 'G'),
 ('HH', 'sta', 'CH'),
 ('HH', 'sta', 'JH'),
 ('HH', 'sta', 'F'),
 ('HH', 'sta', 'S'),
 ('HH', 'sta', 'TH'),
 ('HH', 'sta', 'SH'),
 ('HH', 'sta', 'HH'),
 ('HH', 'sta', 'V'),
 ('HH', 'sta', 'Z'),
 ('HH', 'sta', 'DH'),
 ('HH', 'sta', 'ZH'),
 ('HH', 'sta', 'M'),
 ('HH', 'sta', 'N'),
 ('HH', 'sta', 'NG'),
 ('HH', 'sta', 'L'),
 ('HH', 'sta', 'R'),
 ('HH', 'sta', 'Y'),
 ('HH', 'sta', 'W'),
 ('HH', 'sta', 'IY'),
 ('HH', 'sta', 'IH'),
 ('HH', 'sta', 'UW'),
 ('HH', 'sta', 'UH'),
 ('HH', 'sta', 'EH'),
 ('HH', 'sta', 'ER'),
 ('HH', 'sta', 'AO'),
 ('HH', 'sta', 'AE'),
 ('HH', 'sta', 'AH'),
 ('HH', 'sta', 'AA'),
 ('HH', 'sta', 'EY'),
 ('HH', 'sta', 'OW'),
 ('HH', 'sta', 'OY'),
 ('HH', 'sta', 'AY'),
 ('HH', 'sta', 'AW'),
 ('HH', 'sta', 'stp'),
 ('V', 'sta', 'sta'),
 ('V', 'sta', 'sil'),
 ('V', 'sta', 'spn'),
 ('V', 'sta', 'P'),
 ('V', 'sta', 'T'),
 ('V', 'sta', 'K'),
 ('V', 'sta', 'B'),
 ('V', 'sta', 'D'),
 ('V', 'sta', 'G'),
 ('V', 'sta', 'CH'),
 ('V', 'sta', 'JH'),
 ('V', 'sta', 'F'),
 ('V', 'sta', 'S'),
 ('V', 'sta', 'TH'),
 ('V', 'sta', 'SH'),
 ('V', 'sta', 'HH'),
 ('V', 'sta', 'V'),
 ('V', 'sta', 'Z'),
 ('V', 'sta', 'DH'),
 ('V', 'sta', 'ZH'),
 ('V', 'sta', 'M'),
 ('V', 'sta', 'N'),
 ('V', 'sta', 'NG'),
 ('V', 'sta', 'L'),
 ('V', 'sta', 'R'),
 ('V', 'sta', 'Y'),
 ('V', 'sta', 'W'),
 ('V', 'sta', 'IY'),
 ('V', 'sta', 'IH'),
 ('V', 'sta', 'UW'),
 ('V', 'sta', 'UH'),
 ('V', 'sta', 'EH'),
 ('V', 'sta', 'ER'),
 ('V', 'sta', 'AO'),
 ('V', 'sta', 'AE'),
 ('V', 'sta', 'AH'),
 ('V', 'sta', 'AA'),
 ('V', 'sta', 'EY'),
 ('V', 'sta', 'OW'),
 ('V', 'sta', 'OY'),
 ('V', 'sta', 'AY'),
 ('V', 'sta', 'AW'),
 ('V', 'sta', 'stp'),
 ('Z', 'sta', 'sta'),
 ('Z', 'sta', 'sil'),
 ('Z', 'sta', 'spn'),
 ('Z', 'sta', 'P'),
 ('Z', 'sta', 'T'),
 ('Z', 'sta', 'K'),
 ('Z', 'sta', 'B'),
 ('Z', 'sta', 'D'),
 ('Z', 'sta', 'G'),
 ('Z', 'sta', 'CH'),
 ('Z', 'sta', 'JH'),
 ('Z', 'sta', 'F'),
 ('Z', 'sta', 'S'),
 ('Z', 'sta', 'TH'),
 ('Z', 'sta', 'SH'),
 ('Z', 'sta', 'HH'),
 ('Z', 'sta', 'V'),
 ('Z', 'sta', 'Z'),
 ('Z', 'sta', 'DH'),
 ('Z', 'sta', 'ZH'),
 ('Z', 'sta', 'M'),
 ('Z', 'sta', 'N'),
 ('Z', 'sta', 'NG'),
 ('Z', 'sta', 'L'),
 ('Z', 'sta', 'R'),
 ('Z', 'sta', 'Y'),
 ('Z', 'sta', 'W'),
 ('Z', 'sta', 'IY'),
 ('Z', 'sta', 'IH'),
 ('Z', 'sta', 'UW'),
 ('Z', 'sta', 'UH'),
 ('Z', 'sta', 'EH'),
 ('Z', 'sta', 'ER'),
 ('Z', 'sta', 'AO'),
 ('Z', 'sta', 'AE'),
 ('Z', 'sta', 'AH'),
 ('Z', 'sta', 'AA'),
 ('Z', 'sta', 'EY'),
 ('Z', 'sta', 'OW'),
 ('Z', 'sta', 'OY'),
 ('Z', 'sta', 'AY'),
 ('Z', 'sta', 'AW'),
 ('Z', 'sta', 'stp'),
 ('DH', 'sta', 'sta'),
 ('DH', 'sta', 'sil'),
 ('DH', 'sta', 'spn'),
 ('DH', 'sta', 'P'),
 ('DH', 'sta', 'T'),
 ('DH', 'sta', 'K'),
 ('DH', 'sta', 'B'),
 ('DH', 'sta', 'D'),
 ('DH', 'sta', 'G'),
 ('DH', 'sta', 'CH'),
 ('DH', 'sta', 'JH'),
 ('DH', 'sta', 'F'),
 ('DH', 'sta', 'S'),
 ('DH', 'sta', 'TH'),
 ('DH', 'sta', 'SH'),
 ('DH', 'sta', 'HH'),
 ('DH', 'sta', 'V'),
 ('DH', 'sta', 'Z'),
 ('DH', 'sta', 'DH'),
 ('DH', 'sta', 'ZH'),
 ('DH', 'sta', 'M'),
 ('DH', 'sta', 'N'),
 ('DH', 'sta', 'NG'),
 ('DH', 'sta', 'L'),
 ('DH', 'sta', 'R'),
 ('DH', 'sta', 'Y'),
 ('DH', 'sta', 'W'),
 ('DH', 'sta', 'IY'),
 ('DH', 'sta', 'IH'),
 ('DH', 'sta', 'UW'),
 ('DH', 'sta', 'UH'),
 ('DH', 'sta', 'EH'),
 ('DH', 'sta', 'ER'),
 ('DH', 'sta', 'AO'),
 ('DH', 'sta', 'AE'),
 ('DH', 'sta', 'AH'),
 ('DH', 'sta', 'AA'),
 ('DH', 'sta', 'EY'),
 ('DH', 'sta', 'OW'),
 ('DH', 'sta', 'OY'),
 ('DH', 'sta', 'AY'),
 ('DH', 'sta', 'AW'),
 ('DH', 'sta', 'stp'),
 ('ZH', 'sta', 'sta'),
 ('ZH', 'sta', 'sil'),
 ('ZH', 'sta', 'spn'),
 ('ZH', 'sta', 'P'),
 ('ZH', 'sta', 'T'),
 ('ZH', 'sta', 'K'),
 ('ZH', 'sta', 'B'),
 ('ZH', 'sta', 'D'),
 ('ZH', 'sta', 'G'),
 ('ZH', 'sta', 'CH'),
 ('ZH', 'sta', 'JH'),
 ('ZH', 'sta', 'F'),
 ('ZH', 'sta', 'S'),
 ('ZH', 'sta', 'TH'),
 ('ZH', 'sta', 'SH'),
 ('ZH', 'sta', 'HH'),
 ('ZH', 'sta', 'V'),
 ('ZH', 'sta', 'Z'),
 ('ZH', 'sta', 'DH'),
 ('ZH', 'sta', 'ZH'),
 ('ZH', 'sta', 'M'),
 ('ZH', 'sta', 'N'),
 ('ZH', 'sta', 'NG'),
 ('ZH', 'sta', 'L'),
 ('ZH', 'sta', 'R'),
 ('ZH', 'sta', 'Y'),
 ('ZH', 'sta', 'W'),
 ('ZH', 'sta', 'IY'),
 ('ZH', 'sta', 'IH'),
 ('ZH', 'sta', 'UW'),
 ('ZH', 'sta', 'UH'),
 ('ZH', 'sta', 'EH'),
 ('ZH', 'sta', 'ER'),
 ('ZH', 'sta', 'AO'),
 ('ZH', 'sta', 'AE'),
 ('ZH', 'sta', 'AH'),
 ('ZH', 'sta', 'AA'),
 ('ZH', 'sta', 'EY'),
 ('ZH', 'sta', 'OW'),
 ('ZH', 'sta', 'OY'),
 ('ZH', 'sta', 'AY'),
 ('ZH', 'sta', 'AW'),
 ('ZH', 'sta', 'stp'),
 ('M', 'sta', 'sta'),
 ('M', 'sta', 'sil'),
 ('M', 'sta', 'spn'),
 ('M', 'sta', 'P'),
 ('M', 'sta', 'T'),
 ('M', 'sta', 'K'),
 ('M', 'sta', 'B'),
 ('M', 'sta', 'D'),
 ('M', 'sta', 'G'),
 ('M', 'sta', 'CH'),
 ('M', 'sta', 'JH'),
 ('M', 'sta', 'F'),
 ('M', 'sta', 'S'),
 ('M', 'sta', 'TH'),
 ('M', 'sta', 'SH'),
 ('M', 'sta', 'HH'),
 ('M', 'sta', 'V'),
 ('M', 'sta', 'Z'),
 ('M', 'sta', 'DH'),
 ('M', 'sta', 'ZH'),
 ('M', 'sta', 'M'),
 ('M', 'sta', 'N'),
 ('M', 'sta', 'NG'),
 ('M', 'sta', 'L'),
 ('M', 'sta', 'R'),
 ('M', 'sta', 'Y'),
 ('M', 'sta', 'W'),
 ('M', 'sta', 'IY'),
 ('M', 'sta', 'IH'),
 ('M', 'sta', 'UW'),
 ('M', 'sta', 'UH'),
 ('M', 'sta', 'EH'),
 ('M', 'sta', 'ER'),
 ('M', 'sta', 'AO'),
 ('M', 'sta', 'AE'),
 ('M', 'sta', 'AH'),
 ('M', 'sta', 'AA'),
 ('M', 'sta', 'EY'),
 ('M', 'sta', 'OW'),
 ('M', 'sta', 'OY'),
 ('M', 'sta', 'AY'),
 ('M', 'sta', 'AW'),
 ('M', 'sta', 'stp'),
 ('N', 'sta', 'sta'),
 ('N', 'sta', 'sil'),
 ('N', 'sta', 'spn'),
 ('N', 'sta', 'P'),
 ('N', 'sta', 'T'),
 ('N', 'sta', 'K'),
 ('N', 'sta', 'B'),
 ('N', 'sta', 'D'),
 ('N', 'sta', 'G'),
 ('N', 'sta', 'CH'),
 ('N', 'sta', 'JH'),
 ('N', 'sta', 'F'),
 ('N', 'sta', 'S'),
 ('N', 'sta', 'TH'),
 ('N', 'sta', 'SH'),
 ('N', 'sta', 'HH'),
 ('N', 'sta', 'V'),
 ('N', 'sta', 'Z'),
 ('N', 'sta', 'DH'),
 ('N', 'sta', 'ZH'),
 ('N', 'sta', 'M'),
 ('N', 'sta', 'N'),
 ('N', 'sta', 'NG'),
 ('N', 'sta', 'L'),
 ('N', 'sta', 'R'),
 ('N', 'sta', 'Y'),
 ('N', 'sta', 'W'),
 ('N', 'sta', 'IY'),
 ('N', 'sta', 'IH'),
 ('N', 'sta', 'UW'),
 ('N', 'sta', 'UH'),
 ('N', 'sta', 'EH'),
 ('N', 'sta', 'ER'),
 ('N', 'sta', 'AO'),
 ('N', 'sta', 'AE'),
 ('N', 'sta', 'AH'),
 ('N', 'sta', 'AA'),
 ('N', 'sta', 'EY'),
 ('N', 'sta', 'OW'),
 ('N', 'sta', 'OY'),
 ('N', 'sta', 'AY'),
 ('N', 'sta', 'AW'),
 ('N', 'sta', 'stp'),
 ('NG', 'sta', 'sta'),
 ('NG', 'sta', 'sil'),
 ('NG', 'sta', 'spn'),
 ('NG', 'sta', 'P'),
 ('NG', 'sta', 'T'),
 ('NG', 'sta', 'K'),
 ('NG', 'sta', 'B'),
 ('NG', 'sta', 'D'),
 ('NG', 'sta', 'G'),
 ('NG', 'sta', 'CH'),
 ('NG', 'sta', 'JH'),
 ('NG', 'sta', 'F'),
 ('NG', 'sta', 'S'),
 ('NG', 'sta', 'TH'),
 ('NG', 'sta', 'SH'),
 ('NG', 'sta', 'HH'),
 ('NG', 'sta', 'V'),
 ('NG', 'sta', 'Z'),
 ('NG', 'sta', 'DH'),
 ('NG', 'sta', 'ZH'),
 ('NG', 'sta', 'M'),
 ('NG', 'sta', 'N'),
 ('NG', 'sta', 'NG'),
 ('NG', 'sta', 'L'),
 ('NG', 'sta', 'R'),
 ('NG', 'sta', 'Y'),
 ('NG', 'sta', 'W'),
 ('NG', 'sta', 'IY'),
 ('NG', 'sta', 'IH'),
 ('NG', 'sta', 'UW'),
 ('NG', 'sta', 'UH'),
 ('NG', 'sta', 'EH'),
 ('NG', 'sta', 'ER'),
 ('NG', 'sta', 'AO'),
 ('NG', 'sta', 'AE'),
 ('NG', 'sta', 'AH'),
 ('NG', 'sta', 'AA'),
 ('NG', 'sta', 'EY'),
 ('NG', 'sta', 'OW'),
 ('NG', 'sta', 'OY'),
 ('NG', 'sta', 'AY'),
 ('NG', 'sta', 'AW'),
 ('NG', 'sta', 'stp'),
 ('L', 'sta', 'sta'),
 ('L', 'sta', 'sil'),
 ('L', 'sta', 'spn'),
 ('L', 'sta', 'P'),
 ('L', 'sta', 'T'),
 ('L', 'sta', 'K'),
 ('L', 'sta', 'B'),
 ('L', 'sta', 'D'),
 ('L', 'sta', 'G'),
 ('L', 'sta', 'CH'),
 ('L', 'sta', 'JH'),
 ...]

In [14]: zzz
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[14], line 1
----> 1 zzz

NameError: name 'zzz' is not defined

In [15]: z
Out[15]:
['sta',
 'sil',
 'spn',
 'P',
 'T',
 'K',
 'B',
 'D',
 'G',
 'CH',
 'JH',
 'F',
 'S',
 'TH',
 'SH',
 'HH',
 'V',
 'Z',
 'DH',
 'ZH',
 'M',
 'N',
 'NG',
 'L',
 'R',
 'Y',
 'W',
 'IY',
 'IH',
 'UW',
 'UH',
 'EH',
 'ER',
 'AO',
 'AE',
 'AH',
 'AA',
 'EY',
 'OW',
 'OY',
 'AY',
 'AW',
 'stp']

In [16]: z = ['sta', 'stp',  *dt.phn.tolist()]

In [17]: # zz = [(pre, mid, end) for mid in z for pre in z for end in z]

In [18]: jj = ['XXX', 'XXX', *dt.cls.tolist()]

In [19]: zz = ['-'.join([pre, mid, end]) + '\t' + clcl + '\t'  for mid, clcl in zip(z, jj) for pre in z for end in z]

In [20]: zzz = [zzitem + f"{myidx+1}" for myidx, zzitem in enumerate(zz)]

In [21]: with open('c:/Users/ChienChengChen/Desktop/kk.tsv', 'w') as fout:
    ...:     write('\n'.join(zzz))
    ...:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[21], line 2
      1 with open('c:/Users/ChienChengChen/Desktop/kk.tsv', 'w') as fout:
----> 2     write('\n'.join(zzz))

NameError: name 'write' is not defined

In [22]: with open('c:/Users/ChienChengChen/Desktop/kk.tsv', 'w') as fout:
    ...:     fout.write('\n'.join(zzz))
    ...:

In [23]: 43**3
Out[23]: 79507

In [24]: exit()
21:14:24 unit_demo > 
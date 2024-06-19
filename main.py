import streamlit as st
import pandas as pd
import numpy as np

# load data
# df = pd.read_csv('data/iris.csv')
@st.cache_data 
def loaddata():
    # df = pd.read_csv(r"C:\\Users\\ChienChengChen\\Desktop\\More\\hubert_100__p_xy.tsv", sep="\t")
    df = pd.read_csv("hubert_100__p_xy.tsv", sep="\t")
    return df
df = loaddata()
total_frames = eval(df.columns[0])
# rename col as "phn"
df = df.rename(columns={df.columns[0]: "phn"})

idxmapper = {
    'P':    1,
    'B':    2,
    'T':    3,
    'D':    4,
    'K':    5,
    'G':    6,
    'CH':   7,
    'JH':   8,
    'F':    9,
    'V':   10,
    'S':   11,
    'Z':   12,
    'TH':  13,
    'DH':  14,
    'SH':  15,
    'ZH':  16,
    'HH':  17,
    'M':   18,
    'N':   19,
    'NG':  20,
    'Y':   21,
    'L':   22,
    'R':   23,
    'W':   24,
    'AA':  25,
    'AE':  26,
    'AH':  27,
    'AO':  28,
    'EH':  29,
    'ER':  30,
    'EY':  31,
    'IH':  32,
    'IY':  33,
    'OW':  34,
    'UH':  35,
    'UW':  36,
    'AY':  37,
    'AW':  38,
    'OY':  39,
    'sil': 40,
    'spn': 41,
}

typemapper = {
    'P':   'PLO',
    'B':   'PLO',
    'T':   'PLO',
    'D':   'PLO',
    'K':   'PLO',
    'G':   'PLO',
    'CH':  'AFF',
    'JH':  'AFF',
    'F':   'FRI',
    'V':   'FRI',
    'S':   'FRI',
    'Z':   'FRI',
    'TH':  'FRI',
    'DH':  'FRI',
    'SH':  'FRI',
    'ZH':  'FRI',
    'HH':  'FRI',
    'M':   'NAS',
    'N':   'NAS',
    'NG':  'NAS',
    'Y':   'APP',
    'L':   'APP',
    'R':   'APP',
    'W':   'APP',
    'AA':  'VOW',
    'AE':  'VOW',
    'AH':  'VOW',
    'AO':  'VOW',
    'EH':  'VOW',
    'ER':  'VOW',
    'EY':  'VOW',
    'IH':  'VOW',
    'IY':  'VOW',
    'OW':  'VOW',
    'UH':  'VOW',
    'UW':  'VOW',
    'AY':  'DIP',
    'AW':  'DIP',
    'OY':  'DIP',
    'sil': 'XXX',
    'spn': 'XXX',
}
# print(idxmapper)
if 78-78: st.title('My dataset')
# st.DataFrame(df)
df = df.T.sort_index()
df = df.T
df['phnidx'] = df["phn"].map(lambda i: idxmapper[i.strip()])
# print(df[0][0])
# print(df.columns[-2:])
df = df[list(df.columns[-2:])+
list(df.columns[:-2])]
df = df.sort_values('phnidx')
# 19 	625     	28,601 	265   	428
# 7  	11,505  	2,428  	49    	 78
# 28 	27,699  	95,126 	1,232 	669
# 13 	126,789 	6,947  	1,114 	205
# 57 	558     	46,408 	159   	168

# set phnidx as row index
df = df.set_index('phn')
# drop phnidx column
df = df.drop(columns=['phnidx'])
for col in df.columns:
    df[col] = pd.to_numeric(df[col])
# print(df.columns)
# get max of each column
# max_values = df.max()
# print(max_values)
# st.dataframe(df)
assert (df.values.sum()) == total_frames, "Total frames mismatch"


# Calculate max and argmax for columns
col_max = df.max(axis=0)
col_argmax = df.idxmax(axis=0)
col_sum = df.sum(axis=0)


# Calculate max and argmax for rows
row_max = df.max(axis=1)
row_argmax = df.idxmax(axis=1)
row_sum = df.sum(axis=1)

d = st.dataframe

df_all = df.copy()
df_all.index.name = "phn\\unit"
if 0:d(df_all)
dfrow = (pd.concat([row_sum, row_max, row_argmax, np.round(row_sum/total_frames , 4),
np.round(row_max/row_sum , 4),

], axis=1))
dfrow.columns = ['sum', 'max', 'argmax', 'Pr(phn) '
# '(%)'
,
'Pur (u | p) '
# '(%)'
,
]
if 0:d(dfrow)

dfcol = (pd.concat([col_sum, col_max, col_argmax, np.round(col_sum/total_frames , 4),
np.round(col_max/col_sum , 4),
col_argmax.map(lambda i: idxmapper[i.strip()]),
], axis=1))
dfcol.columns = ['sum', 'max', 'argmax', 'Pr(unit) '
# '(%)'
,
'Pur (p | u) '
# '(%)'
, 'phnidx',
]
dfcol = dfcol.sort_values('sum', ascending=False)
dfcol = dfcol.sort_values('phnidx', ascending=True, kind='mergesort')
dfcol = dfcol.drop(columns=['phnidx'])
# rename index name
dfcol.index.name = 'unit'
if 0:d(dfcol)
if 0:st.write("Total: {}".format( (row_sum/total_frames).sum()))

# d(df_all)[dfcol['unit']]
newdf = df_all.loc[:, dfcol.index] / total_frames
if 88-88: st.write("newdf: {}".format(newdf.sum().sum()))
if 1-1:d(
    newdf
)
if 1-1:d(dfcol['Pr(unit) '
# '(%)'
].T)

import matplotlib.pyplot as plt
import seaborn as sns


def myplotter(pp, ha=False, va=False, hb=None, vb=None,):



    if 1:
        if 1:
            # 分组信息
            if hb is None:
                group_sizes = [1,1,1,1,1,
                            ]
                group_sizes.append(pp.shape[1] - sum(group_sizes))
            else:
                group_sizes = hb
            group_positions = np.cumsum(group_sizes)[:-1]
            if vb is None:

                hgroup_sizes = [6,2,9,3,4,12,3]
                hgroup_sizes.append(pp.shape[0] - sum(hgroup_sizes))
            else:
                hgroup_sizes = vb
            hgroup_positions = np.cumsum(hgroup_sizes)[:-1]
            
    fig, ax = plt.subplots()
    ratio = 0.25
    fig.set_size_inches(int(round(100*ratio)), int(round(41*ratio)))
    axi = sns.heatmap(pp, ax=ax, cmap="YlGnBu", 
    # font
    # fontsize=8,
    # annot=True, fmt=".2f"
    )
    # axi.set_xticklabels(axi.get_xticklabels(), rotation=45)
    axi.set_yticks(np.arange(pp.shape[0]) + 0.5, minor=False)
    axi.set_yticklabels(
        pp.index, rotation=0, fontsize=12,
        # each 1
        minor=False,
        )
    axi.set_xticks(np.arange(pp.shape[1]) + 0.5, minor=False)
    axi.set_xticklabels(
        pp.columns, rotation=75, fontsize=12,
        # each 1
        minor=False,
        )


    # st.write(pp.index)
    # axi.set_yticks(list(pp.index))
    # print(axi.get_yticklabels())
    # axi.set_yticklabels(
    #     axi.get_yticklabels(), rotation=45, fontsize=8,
    #     # each 1
    #     minor=True,
    #     
    #     )
    if 1:
        if 1:
            # 绘制分割线
            for pos in group_positions:
                if ha:
                    ax.axvline(x=pos, color='red', linestyle='-', linewidth=1)
                else:
                    ax.axvline(x=pos, color='blue', linestyle='--', linewidth=0.5)

            for pos in hgroup_positions:
                if va:
                    ax.axhline(y=pos, color='red', linestyle='-', linewidth=1)
                else:
                    ax.axhline(y=pos, color='blue', linestyle='--', linewidth=0.5)
    if 2:st.pyplot(fig)



# p_up = p(u | p) * p(p)
# p_up = p(p | u) * p(u)
# p(p|u) = p_up / p(u)
v1 = (newdf) / (dfcol['Pr(unit) '
# '(%)'
])
v1.index.name = 'Pr(p | u) '
# '(%)'


if 99-99:d(
    
    np.round(v1 * 100, 2)
)

# matplot 
import matplotlib.pyplot as plt
import seaborn as sns
# sns.heatmap(v1)
# st.seaborn.heatmap(v1)

# v2 = newdf / 
# d(dfrow['Pr(phn) ']
# )
# st.write(newdf.shape)
# st.write(dfrow['Pr(phn) '].shape)
# st.write((newdf.T/dfrow['Pr(phn) ']).T.shape)
v2 = (newdf.T/dfrow['Pr(phn) ']).T

v2.index.name = 'Pr(u | p) '
# '(%)'

if 99-99:d(
    np.round(v2 * 100, 2)
)
# print(df.values / total_frames)



# st.write("Total: {}".format( (v2).sum().sum()))


# streamlit selectbox

idx2 = st.selectbox(
    "Select visualization",
    [
        "Pr_(p, u)",
        "Pr(p | u)",
        "Pr(u | p)",
    ],
)



typeorder = ["PLO","AFF","FRI","NAS","APP","VOW","DIP","XXX",]
dbetter = dfcol["argmax"].to_frame()
# st.write(dbetter.to_frame())
dbetter['phoneclass'] = dfcol["argmax"].map(lambda i: typemapper[i.strip()])
# st.write(dbetter)
rescnt = (dbetter['phoneclass'].value_counts()[typeorder])
rescnt = rescnt.values
# st.write(rescnt)

if idx2 == "Pr_(p, u)":
    st.markdown("#### Joint Prob")
    myplotter(newdf       , 
              hb=rescnt
              )
    ## pmi = np.log(p_xy / np.matmul(p_x, p_y) + 1e-8)
    ## mi = (p_xy * pmi).sum()
    # st.write("MI: {}".format(
    #     (newdf * np.log(newdf / np.matmul(newdf.sum(axis=1)[:, None], newdf.sum(axis=0)[None, :]) + 1e-8)).sum()
    # ))
    p_x = newdf.values.sum(axis=1)[:, None]
    p_y = newdf.values.sum(axis=0)[None]
    p_xy = newdf.values
    pmi = np.log(p_xy / np.matmul(p_x, p_y) + 1e-8)
    mi = (p_xy * pmi).sum()
    st.write("MI: {}".format(mi))
    # st.write(newdf.sum().sum())
    axqx=int(st.selectbox('Axis', ['row_first', 'col_first']) == 'row_first')
    vv = newdf.sum(axis=axqx)
    if 0:st.write(
        vv
    )
    # matplotlib bar plot
    plt.clf()
    grapap = plt.figure(figsize=(10, 5))
    plt.bar(vv.index, vv.values)
    def myentropy(p):
        return -np.sum(p * np.log(p + 1e-10))
    st.write("Phoneme Entropy: {}".format(myentropy(vv.values)))
    # st.write(col_max.sum() / total_frames)
    st.write("Phoneme Purity: {}".format(col_max.sum() / total_frames)) 
    # st.write(vv.values)
    plt.xticks(rotation=75, fontsize=8)
    st.pyplot(grapap)

    vv = newdf.sum(axis=1-axqx)
    if 0:st.write(
        vv
    )
    # matplotlib bar plot
    plt.clf()
    grapap = plt.figure(figsize=(10, 5))
    plt.bar(vv.index, vv.values)
    plt.xticks(rotation=75, fontsize=8)
    st.write("Unit Entropy: {}".format(myentropy(vv.values)))
    st.write("Unit Purity: {}".format(row_max.sum() / total_frames))
    st.pyplot(grapap)

    # slice a column to plot
    SSout = (newdf.columns)
    idx001 = st.selectbox('Select a column', SSout)
    # newdf = newdf.sort_values(by=idx001, ascending=False)
    ### st.write(idx001)
    ### st.write(newdf)
    ### st.write(newdf.columns)
    slicedout = newdf.iloc[:, list(SSout).index(idx001)]
    if 4-4:st.write(
        slicedout
    )
    # matplotlib bar plot
    plt.clf()

    grapap = plt.figure(figsize=(10, 5))
    plt.bar(slicedout.index, slicedout.values)
    plt.xticks(rotation=75, fontsize=8)
    plt.title(idx001)
    st.pyplot(grapap)
    # entropy
    p_mar = slicedout.values / (slicedout.values.sum())
    # st.write(p_mar)
    # st.write(p_mar.sum())
    st.write("Entropy: {}".format(myentropy(p_mar)))

    # slice a column to plot
    newdf2 = newdf.T
    SSout2 = (newdf2.columns)
    idx001 = st.selectbox('Select a row', SSout2)
    # newdf2 = newdf2.sort_values(by=idx001, ascending=False)
    ### st.write(idx001)
    ### st.write(newdf2)
    ### st.write(newdf2.columns)
    slicedout = newdf2.iloc[:, list(SSout2).index(idx001)]
    if 4-4:st.write(
        slicedout
    )
    # matplotlib bar plot
    plt.clf()
    grapap = plt.figure(figsize=(10, 5))
    plt.bar(slicedout.index, slicedout.values)
    plt.xticks(rotation=75, fontsize=8)
    plt.title(idx001)
    st.pyplot(grapap)
    # entropy
    p_mar = slicedout.values / (slicedout.values.sum())
    # st.write(p_mar)
    # st.write(p_mar.sum())
    st.write("Entropy: {}".format(myentropy(p_mar)))

elif idx2 == "Pr(p | u)":
    st.markdown("#### Pr(p | u) 每個 column 總和為 100 %")
    myplotter(v1, ha=True, hb=rescnt)
    # st.write(v1.sum().sum())
elif idx2 == "Pr(u | p)":
    st.markdown("#### Pr(u | p) 每個 row 總和為 100 %")
    myplotter(v2, va=True, hb=rescnt)
    # st.write(v2.sum().sum())

tabel_optinos, tabres = zip(*[
    ("??", df),
    ("??2", df_all),
    ("Pr_(p, u)", newdf),
    ("Pr(p | u)", v1),
    ("Pr(u | p)", v2),
    ("ResPr(p | u)", dfrow),
    ("ResPr(u | p)", dfcol),
])
idx3 = st.selectbox(
    "Select table",
    tabel_optinos,
)
if 1:d(tabres[tabel_optinos.index(idx3)])

if 0:idx1 = st.selectbox('Select a column', df.columns)

#########3 entropy
#########3 wnhsu code

def entropy(p):
    return -np.sum(p * np.log(p + 1e-10))



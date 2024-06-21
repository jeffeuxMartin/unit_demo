# TODO: purity table at top!
# URL 帶參數
# 深淺 plot
import streamlit as st
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def loaddata(modeltype='hubert', clu=100):
    # data = np.load("C:\\Users\\ChienChengChen\\Desktop\\mymeasure\\analysis_data\\{}\\clu{:03d}\\train-clean-100.npz".format(modeltype, clu) )
    data = np.load("analysis_data/{}/clu{:03d}/train-clean-100.npz".format(modeltype, clu) )
    
    return data

urlmodeltype = st.query_params.get('modeltype', 'hubert')
inmymodeltype = st.text_input('modeltype', urlmodeltype)
if inmymodeltype != urlmodeltype:
    st.query_params['modeltype']=inmymodeltype
    urlmodeltype = inmymodeltype
mymodeltype = urlmodeltype

urlclu = st.query_params.get('clu', '100')
inmyclu = st.text_input('clu', urlclu)
if inmyclu != urlclu:
    st.query_params['clu']=inmyclu
    urlclu = inmyclu
clu = int(urlclu)
data = loaddata(mymodeltype, clu)

st.markdown("## {}".format(mymodeltype))
knowledge = pd.read_csv(
    './knowledge_table.tsv',
    sep='\t',
)

# st.write(knowledge)

# st.write(
#     list(data.keys())
# )

def plot_heatmap(pp, gropui=None, hb=None, vb=None, ha=True, va=True, annot=False, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    ratio = 0.25
    fig.set_size_inches(int(round(100*ratio)), int(round(41*ratio)))
    optionals = {}
    if vmin is not None:
        optionals['vmin'] = vmin
    if vmax is not None:
        optionals['vmax'] = vmax
    axi = sns.heatmap(pp, ax=ax, cmap="YlGnBu", 
                      **optionals,
    )
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
    pass
    if gropui is not None:
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
        if 1:
            if 1:
                if hb is not None:
                    # 绘制分割线
                    for pos in group_positions:
                        if ha:
                            ax.axvline(x=pos, color='red', linestyle='-', linewidth=1)
                        else:
                            ax.axvline(x=pos, color='blue', linestyle='--', linewidth=0.5)
                if vb is not None:
                    for pos in hgroup_positions:
                        if va:
                            ax.axhline(y=pos, color='red', linestyle='-', linewidth=1)
                        else:
                            ax.axhline(y=pos, color='blue', linestyle='--', linewidth=0.5)  
    if annot:
        for i in range(pp.shape[0]):
            for j in range(pp.shape[1]):
                ax.text(j+0.5, i+0.5, '%.2f %%' % (pp.iloc[i, j] * 100),
                        ha='center', va='center', color='red', fontsize=18)
    st.pyplot(fig)
    # close figure
    plt.close(fig)

# plot_heatmap(data.get('p_xy'))
# st.write(data.get('p_xy'))
mydata = data.get('p_xy')
if 0: mydata = data.get('p_x__given__y')
if 0: mydata = data.get('p_y__given__x')
# st.write(data.get('ref2pid'))
# ref2pid = (data.get('ref2pid'))
# ref2pid0 = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'B': 6, 'CH': 7, 'D': 8, 'DH': 9, 'EH': 10, 'ER': 11, 'EY': 12, 'F': 13, 'G': 14, 'HH': 15, 'IH': 16, 'IY': 17, 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23, 'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29, 'T': 30, 'TH': 31, 'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38, 'sil': 39, 'spn': 40}
# ref2pid = list(ref2pid0.keys())
# st.write(ref2pid)
# for ij, it in enumerate(ref2pid):
#     assert ref2pid0[it] == ij
# st.write(data.get('hyp2lid'))
hyp2lid = (data.get('hyp2lid'))
ref2pid = (data.get('ref2pid'))

df = pd.DataFrame(
    mydata,
    index=ref2pid,
    columns=hyp2lid,
)

# st.dataframe(df.T)
# st.data_editor?
# plot_heatmap(df)
# print(df)
# plot_heatmap(df)

# st.dataframe(df)
# st.write(
#     df.max()
# )

if "entropy":
    entropy_phn = data.get('h_each_x')

    df['entropy_phn'] = entropy_phn
    # st.write(df)
    df = df.sort_values(by='entropy_phn', ascending=False)
    df = df.drop(columns=['entropy_phn'])

    entropy_unit = data.get('h_each_y')
    # st.write(entropy_unit.shape)
    # st.write(entropy_unit)
    df = df.T
    df['entropy_unit'] = entropy_unit
    df = df.sort_values(by='entropy_unit', ascending=False)
    df = df.drop(columns=['entropy_unit'])
    df = df.T


    if 0+0: plot_heatmap(df)

if 0:
    odf = (df.T[[ 'P', 'B', 'T', 'D', 'K', 'G', 'CH', 'JH', 'F', 'V', 'S', 'Z', 'TH', 'DH', 'SH', 'ZH', 'HH', 'M', 'N', 'NG', 'Y', 'L', 'R', 'W', 'AA', 'AE', 'AH', 'AO', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'UH', 'UW', 'AY', 'AW', 'OY', 'sil', 'spn',]].T)
    odf.columns = [eval(it) for it in odf.columns]
    
    odf = odf.T.sort_index(ascending=True, inplace=False).T
    plot_heatmap(odf)


# st.write(df)
# st.write(df.T[knowledge['phn'].to_list()].T)
df1 = df.T[knowledge['phn'].to_list()].T
if 0:plot_heatmap(df1, gropui=True, 
             hb=None,
                vb=[6,2,9,3,4,12,3,2],
             )
df2 = df1.T
# st.write(df2.T)
representative_phn_of_units = (df1.idxmax(axis=0))
max_prob_of_representative_phns = df1.max(axis=0)
representative_phn_of_units_idx = representative_phn_of_units.apply(lambda x: knowledge['phn'].to_list().index(x))
# df_sortedunits = df2.T.sort_values(by=representative_phn_of_units_idx, axis=1, ascending=True)
# st.write(representative_phn_of_units)
# st.write(max_prob_of_representative_phns)
# st.write(representative_phn_of_units_idx)
df_sort_by_aligned_phn_and_prob = pd.concat([representative_phn_of_units, max_prob_of_representative_phns, 
         representative_phn_of_units_idx                                    ], axis=1)
df_sort_by_aligned_phn_and_prob.columns = ['phn', 'prob', 'phn_idx']
df_sort_by_aligned_phn_and_prob['sorter_value'] = (representative_phn_of_units_idx.max()+1-df_sort_by_aligned_phn_and_prob['phn_idx'] + max_prob_of_representative_phns / 100)
df_sort_by_aligned_phn_and_prob = df_sort_by_aligned_phn_and_prob.sort_values(by='sorter_value', ascending=False)
# st.write(df_sort_by_aligned_phn_and_prob)
dfcool = df1.loc[:, df_sort_by_aligned_phn_and_prob.index]
# plot_heatmap(dfcool)
# st.write(df1.loc[:, df_sort_by_aligned_phn_and_prob.index])

grouings_good = df2.idxmax(axis=1).value_counts()
# knowledge[['phn', 'cls']].groupby('phn').count()
phn_to_cls = knowledge[['phn', 'cls']].groupby('phn').first()
# grouings_good group by cls
grouings_good_cls = grouings_good.to_frame()
grouings_good_cls['cls1'] = grouings_good_cls.index.map(lambda x: phn_to_cls.loc[x, 'cls'])
grouings_good_cls = grouings_good_cls.reset_index()
grouings_good_cls['phn'] = grouings_good_cls['index']
grouings_good_cls = grouings_good_cls.drop(columns=['index'])
grouings_good_cls = grouings_good_cls[[
    'cls1',
    'phn',
    'count',
   
]]
# grouings_good_cls = grouings_good_cls.join(phn_to_cls, on='index')
# sum 'count' by 'cls1'

grouings_good_cls = grouings_good_cls.groupby('cls1').aggregate('sum')
# if 0, add 0 to all cls
for it in knowledge['cls'].unique():
    if it not in grouings_good_cls.index:
        grouings_good_cls.loc[it] = 0
# st.write(grouings_good_cls)
mycls = (
    knowledge['cls'].unique().tolist()
)
grouings_good_cls = (grouings_good_cls['count'])
resultgrouping = grouings_good_cls.loc[mycls].values.tolist()
# st.write(resultgrouping.values.tolist())

plot_heatmap(dfcool, gropui=True, 
             hb=resultgrouping,
                vb=[6,2,9,3,4,12,3,2],
                ha=False, va=False,
                vmax=0.025,
                vmin=0.0,
             )

# confusion matrix of sections
print(resultgrouping)
phonetogroup = [6,2,9,3,4,12,3,2]
print(phonetogroup)
probs = dfcool.values

def getconfusionmatrix(probs, resultgrouping, phonetogroup):
    # probs = probs / probs.sum()
    # print(probs.sum())
    # print(probs)
    pass
    print(probs.shape)
    boundaries = lambda x: list(zip(*([0] + x.tolist()[:-1], x.tolist())))
    bbx = (boundaries(np.array(resultgrouping).cumsum()))
    bby = (boundaries(np.array(phonetogroup).cumsum()))
    result = np.zeros((len(resultgrouping), len(phonetogroup)))
    # print(result.shape)
    for ni, bx in enumerate(bbx):
        for nj, by in enumerate(bby):
            # print("ni, nj", ni, nj)
            # print("bx, by", bx, by)
            result[ni, nj] = probs[
                by[0]:by[1],
                bx[0]:bx[1], 
                ].sum()
            # print(result[ni, nj])
    return result
confusionmatrix = getconfusionmatrix(probs, resultgrouping, phonetogroup)
# print(confusionmatrix)
plt.clf()
grapap = plt.figure(figsize=(10, 5))
# plot_heatmap((confusionmatrix)
             
#              )
plot_heatmap(pd.DataFrame(confusionmatrix, index=mycls, columns=mycls),
             annot=True,)

# plot_heatmap(df_sortedunits, gropui=True, 
#              hb=None,
#                 vb=[6,2,9,3,4,12,3,2],
#              )

# st.write(df[knowledge['phn']
#             ])
# plot_heatmap(df[knowledge['phn'].values])
# TODO: sort by entropy
# TODO: sort by purity
# TODO: sort by probability
# TODO: sort by phonetics

# plot barchart of phoneme
plt.clf()
grapap = plt.figure(figsize=(10, 5))
########################
plot_heatmap(df1, gropui=True, 
             hb=None,
                vb=[6,2,9,3,4,12,3,2],
             )
plot_heatmap(df)
## note: 重點不是 sort by! 重點是 橫軸、縱軸要呈現，然後找出 top2 of both axis 畫出長條圖
## 最後是那個 entropy 直方圖！

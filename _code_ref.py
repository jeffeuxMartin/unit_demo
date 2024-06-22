import streamlit as st, matplotlib.pyplot as plt, seaborn as sns
h_each_phn = []
fig = plt.figure(figsize=(10, 5))
sns.histplot(h_each_phn, 
             bins=25,
             kde=True,
             )
# matplotlib font
plt.rcParams['font.sans-serif'] = [
    # 'Microsoft JhengHei'
    # "標楷體"
    # "Noto Sans CJK TC"
    # "Iosevka Jeff"
    "Times New Roman",
    "DFKai-SB",
]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.xlabel('音素熵', fontsize=12, fontdict={'family': 'sans-serif', 'weight': 'bold'})
plt.xlabel('音素熵', fontsize=12, fontdict={'family': 'DFKai-SB', 'weight': 'bold'})
plt.ylabel('音熵', fontsize=12, fontdict={'family': 'Microsoft JhengHei', 'weight': 'bold'})
st.pyplot(fig)
plt.close(fig)

{
0:"DejaVu Sans",
1:"Bitstream Vera Sans",
2:"Computer Modern Sans Serif",
3:"Lucida Grande",
4:"Verdana",
5:"Geneva",
6:"Lucid",
7:"Arial",
8:"Helvetica",
9:"Avant Garde",
10:"sans-serif",
}
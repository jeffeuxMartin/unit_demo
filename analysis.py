import streamlit as st
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from utils import (
    troublesome_尻ㄙㄟ,
    entropy欸尻ㄙㄟ,
)

@st.cache_resource
def load_data(
    model_type: str = "hubert",
    cluster_num: int = 100,
):
    return np.load(
        "./analysis_data/"
       f"{model_type}/"
       f"clu{cluster_num:03d}/"
        "train-clean-100.npz"
    )

def parse_urlargs():
    pass
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
    return mymodeltype, clu

def plot_heatmap(
    df: pd.DataFrame,
    x_groups = None,
    y_groups = None,
    vmin = None,
    vmax = None,
    horizontal_highlighted = False,
    vertical_highlighted = False,
    annotated = False,
  ):
    plt.clf()
    graph = plt.figure(figsize=(25, 10))
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 10)

    options = {}
    if vmin is not None:
        options["vmin"] = vmin
    if vmax is not None:
        options["vmax"] = vmax

    

    axis = sns.heatmap(
        df,
        ax=ax,
        cmap="YlGnBu",
        **options,
    )

    axis.set_xticks(
        np.arange(df.shape[1]) + 0.5,
        minor = False,
    )
    axis.set_xticklabels(
        df.columns,
        rotation=75 if df.shape[1] <= 100 else 90,
        fontsize=12 if df.shape[1] <= 100 else 6,
        # horizontalalignment='right',
        minor = False,  # each 1
    )
    axis.set_yticks(
        np.arange(df.shape[0]) + 0.5,
        minor = False,
    )
    axis.set_yticklabels(
        df.index,
        rotation=0,
        fontsize=12,
        minor = False,  # each 1
    )

    if x_groups is not None:
        for pos in np.cumsum(x_groups)[:-1]:
            if vertical_highlighted:
                axis.axvline(pos, color='r', linewidth=1, linestyle="-")
            else:
                axis.axvline(pos, color='blue', linewidth=0.5, linestyle="--")
    if y_groups is not None:
        for pos in np.cumsum(y_groups)[:-1]:
            if horizontal_highlighted:
                axis.axhline(pos, color='r', linewidth=1, linestyle="-")
            else:
                axis.axhline(pos, color='blue', linewidth=0.5, linestyle="--")
    
    if annotated:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                axis.text(j + 0.5, i + 0.5, 
                          f"{df.iloc[i, j] * 100:.2f} %",
                          ha="center", va="center", 
                          color="red", fontsize=18,
                )
    st.pyplot(fig)
    plt.close(fig)
    return fig

def plot_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
):
    ## TO check --- by Copilot
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 10)

    axis = sns.barplot(
        x=x,
        y=y,
        data=df,
        ax=ax,
    )
    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=75,
        fontsize=12,
        horizontalalignment='right',
    )
    st.pyplot(fig)
    plt.close(fig)
    return fig

def get_confusion_matrix(
    probs: np.ndarray,
    unit_groups: np.ndarray,
    phn_groups: np.ndarray,
):
    unit_groups = np.array(unit_groups)
    phn_groups = np.array(phn_groups)

    def boundaries(x):
        return zip(*([0] + x.tolist()[:-1], x.tolist()))

    result = np.zeros((len(unit_groups), len(phn_groups)))
    for ni, (bx_from, bx_to) in enumerate(
                boundaries(phn_groups.cumsum())):
        for nj, (by_from, by_to) in enumerate(
                boundaries(unit_groups.cumsum())):
            result[ni, nj] = probs[
                bx_from : bx_to, 
                by_from : by_to,
            ].sum()
    return result

if __name__ == "__main__":
    analysis_obj = parse_urlargs()
    data = load_data(*analysis_obj)
    st.markdown(f"## {analysis_obj[0]} with {analysis_obj[1]} clusters")

    knowledge_data = pd.read_csv(
        "./knowledge_table.tsv", sep="\t"
    )

    PHONETIC_GROUPS = (
        knowledge_data['cls'].unique().tolist())
    PHONETIC_GROUP_NUMS = [6, 2, 9, 3, 4, 12, 3, 2]  # FIXME


    df__joint_prob = pd.DataFrame(
        data.get("p_xy"),
        index=data.get("ref2pid"),
        columns=data.get("hyp2lid"),
    )

    ## df__joint_prob = (
    ##     df__joint_prob.div(
    ##         df__joint_prob.sum(axis=1), axis=0))


    dfcool, resultgrouping = troublesome_尻ㄙㄟ(
        df__joint_prob, knowledge_data)

    plot_heatmap(
        dfcool,
        x_groups=resultgrouping,
        y_groups=PHONETIC_GROUP_NUMS,
        vmin=0.0, vmax=0.025,
    )

    confusion_matrix = get_confusion_matrix(
        dfcool.values,
        resultgrouping,
        PHONETIC_GROUP_NUMS,
    )

    plot_heatmap(
        pd.DataFrame(
            confusion_matrix,
            index=PHONETIC_GROUPS,
            columns=PHONETIC_GROUPS,
        ),
        annotated=True,
    )



    df1, df = entropy欸尻ㄙㄟ(
        df__joint_prob,
        entropy_phn = data.get('h_each_x'),
        entropy_unit = data.get('h_each_y'),
        knowledge_data=knowledge_data,
    )

    plot_heatmap(
        df1,
        y_groups=PHONETIC_GROUP_NUMS,
        vmin=0.0, 
        vmax=0.025,
        horizontal_highlighted=True,
    )

    plot_heatmap(
        df,
        vmax=0.025,
    )

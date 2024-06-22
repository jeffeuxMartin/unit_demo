## TODO: 音素熵 distribution
## 70%
## Average number of unique units per phoneme

DEBUG = False
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
    # graph = plt.figure(figsize=(25, 10))
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
    plt.clf()
    # graph = plt.figure(figsize=(10, 5))
    ## TO check --- by Copilot
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 10)

    # plt.bar(ser.index, ser.values, color='blue')
    axis = sns.barplot(
        x=x,
        y=y,
        data=df,
        ax=ax,
    )
    axis.set_xticks(
        np.arange(df.shape[0]),
    )
    axis.set_xticklabels(
        df.index,  # axis.get_xticklabels(),
        rotation=75,
        fontsize=12,
        horizontalalignment='right',
    )
    st.pyplot(fig)
    plt.close(fig)
    return fig

def plot_barplot__series(
    ser: pd.Series,
    toolkit = 'seaborn',
    label = None,
    horizontal = False,
    color = None,
):
    plt.clf()
    # graph = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots()
    fig.set_size_inches(25, 10)
    ser.index.name = label

    if toolkit == 'matplotlib':
        plt.bar(ser.index, ser.values, color=color)
    elif toolkit == 'seaborn':
        if not horizontal:
            sns.barplot(
                x=ser.index,
                y=ser.values,
                ax=ax,
                color=color,
            )
        else:
            sns.barplot(
                y=ser.index,
                x=ser.values,
                ax=ax,
                color=color,                
            )
    else:
        raise ValueError(f"Unknown toolkit: {toolkit}")

    st.pyplot(fig)
    plt.close(fig)
    return fig

def plot_barplot___copilot_series(
    ser: pd.Series,
):
    plot_barplot(
        pd.DataFrame(
            ser,
            columns=['prob'],
            index=ser.index,
        ).reset_index(),
        x='index',
        y='prob',
    )

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

def entropy(p):
    return -np.sum(p * np.log(p + 1e-8))

if __name__ == "__main__":
    analysis_obj = parse_urlargs()
    data = load_data(*analysis_obj)
    st.markdown(f"## {analysis_obj[0]} with {analysis_obj[1]} clusters")

    knowledge_data = pd.read_csv(
        "./knowledge_table.tsv", sep="\t"
    )

    # st.write(knowledge_data)
    lookup_phn_to_cls = dict(
        zip(knowledge_data['phn'], knowledge_data['cls'])
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

    def DEMO_BOTTOM():

        st.write(dfcool)

    phn_purity = dfcool.max(axis=0).sum()
    unit_purity = dfcool.max(axis=1).sum()
    phn_entropy = entropy(dfcool.values.sum(axis=1))
    unit_entropy = entropy(dfcool.values.sum(axis=0))
    mi = entropy(dfcool.values.sum(axis=0)) + entropy(dfcool.values.sum(axis=1)) - entropy(dfcool.values)

    st.dataframe(pd.DataFrame({
        'model_type': analysis_obj[0],
        'phn_purity': phn_purity,
        'unit_purity': unit_purity,
        'phn_entropy': phn_entropy,
        'unit_entropy': unit_entropy,
        'mi': mi,
        'PNMI': mi / phn_entropy,
        'UNMI': mi / unit_entropy,
    }, index=[0]), hide_index=True)

    st.dataframe(pd.DataFrame({
        'ref_segL': data.get('refsegL').tolist(),
        'hyp_segL': data.get('hypsegL').tolist(),
        'p_xy_shape': [data.get('p_xy').shape],
        'frm_tot': data.get('tot'),
        'frm_diff': data.get('frmdiff'),
        'utt_tot': data.get('utt_tot'),
        'utt_miss': data.get('utt_miss'),
    }), hide_index=True)



    h_each_phn = pd.Series(
        data.get('h_each_x'),
        name='h_each_phn',
        index=data.get('ref2pid'),
    )[dfcool.index]

    h_each_unit = pd.Series(
        data.get('h_each_y'),
        name='h_each_unit',
        index=data.get('hyp2lid'),
    )[dfcool.columns]


    if not DEBUG:
        plot_heatmap(
            dfcool,
            x_groups=resultgrouping,
            y_groups=PHONETIC_GROUP_NUMS,
            vmin=0.0, vmax=0.025,
        )

        ##################################
        ## st.write(
        ##     # dfcool.sum(axis=0).sort_values(ascending=False).head(10)
        ## )
        prob_units = dfcool.sum(axis=0)
        prob_phns = dfcool.sum(axis=1)

        plot_barplot__series(
            prob_units,
            label='prob_units',
        )
        plot_barplot__series(
            h_each_unit,
            label='h_each_unit',
            color='red',
        )

        plot_barplot__series(
            prob_phns,
            label='prob_phns',
            horizontal=True,
        )
        plot_barplot__series(
            h_each_phn,
            label='h_each_phn',
            color='red',
            horizontal=True,
        )

    # dfcool as options
    to_visualize_phn = st.selectbox(
        "Select a phoneme to visualize",
        dfcool.index,
    )
    plot_barplot__series(
        dfcool.T[to_visualize_phn],
        label='prob_{}'.format(to_visualize_phn),
    )
    marginal_prob_of_phn = dfcool.T[to_visualize_phn].sum()
    st.write(
        "entropy of the selected phoneme: ",
        entropy(dfcool.T[to_visualize_phn].values / marginal_prob_of_phn),
    )

    to_visualize_unit = st.selectbox(
        "Select a unit to visualize",
        dfcool.columns,
    )
    plot_barplot__series(
        dfcool[to_visualize_unit],
        label='prob_{}'.format(to_visualize_unit),
    )
    marginal_prob_of_unit = dfcool[to_visualize_unit].sum()
    st.write(
        "entropy of the selected unit: ",
        entropy(dfcool[to_visualize_unit].values / marginal_prob_of_unit),
    )

    ##################################
    if not DEBUG:
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


    ##################################
        DEMO_BOTTOM()


    st.markdown("## Prob")
    prob__top_phn = pd.DataFrame({
        'Top_phn_l': prob_phns.sort_values(ascending=False).head(10).index,
        'Top_phn_v': prob_phns.sort_values(ascending=False).head(10).values
    })
    prob__bottom_phn = pd.DataFrame({
        'Bot_phn_l': prob_phns.sort_values(ascending=True).head(10).index,
        'Bot_phn_v': prob_phns.sort_values(ascending=True).head(10).values
    })

    # Create DataFrames for top and bottom 10 of h_each_unit
    prob__top_unit = pd.DataFrame({
        'Top_unit_l': prob_units.sort_values(ascending=False).head(10).index,
        'Top_unit_v': prob_units.sort_values(ascending=False).head(10).values
    })
    prob__bottom_unit = pd.DataFrame({
        'Bot_unit_l': prob_units.sort_values(ascending=True).head(10).index,
        'Bot_unit_v': prob_units.sort_values(ascending=True).head(10).values
    })

    # Concatenate all DataFrames horizontally
    prob_rank = pd.concat([prob__top_phn, prob__bottom_phn, prob__top_unit, prob__bottom_unit], axis=1)
    # start from 1
    prob_rank.index = np.arange(1, prob_rank.shape[0] + 1)
    st.write(prob_rank)

    st.markdown("## Entropy")
    entr__top_phn = pd.DataFrame({
        'Top_phn_l': h_each_phn.sort_values(ascending=False).head(10).index,
        'Top_phn_v': h_each_phn.sort_values(ascending=False).head(10).values
    })
    entr__bottom_phn = pd.DataFrame({
        'Bot_phn_l': h_each_phn.sort_values(ascending=True).head(10).index,
        'Bot_phn_v': h_each_phn.sort_values(ascending=True).head(10).values
    })

    # Create DataFrames for top and bottom 10 of h_each_unit
    entr__top_unit = pd.DataFrame({
        'Top_unit_l': h_each_unit.sort_values(ascending=False).head(10).index,
        'Top_unit_v': h_each_unit.sort_values(ascending=False).head(10).values
    })
    entr__bottom_unit = pd.DataFrame({
        'Bot_unit_l': h_each_unit.sort_values(ascending=True).head(10).index,
        'Bot_unit_v': h_each_unit.sort_values(ascending=True).head(10).values
    })

    # Concatenate all DataFrames horizontally
    entropy_rank = pd.concat([entr__top_phn, entr__bottom_phn, entr__top_unit, entr__bottom_unit], axis=1)
    # start from 1
    entropy_rank.index = np.arange(1, entropy_rank.shape[0] + 1)
    st.write(entropy_rank)


    # combine the above into a single table
    st.write("## 每個 phoneme 的 top 5 units")
    top5_unit = dfcool.T.apply(
        lambda x: x.sort_values(ascending=False).head(5).index.tolist(),
        axis=0,
    )
    # add phonetic group
    top5_unit_T = top5_unit.T
    top5_unit_T['cls'] = top5_unit_T.index.map(lookup_phn_to_cls)
    # move cls to the front
    top5_unit_T = top5_unit_T[['cls'] + [col for col in top5_unit_T.columns if col != 'cls']]
    top5_unit = top5_unit_T.T
    top5_unit_prob = dfcool.T.apply(
        lambda x: x.sort_values(ascending=False).head(5).values.tolist(),
        axis=0,
    )

    st.write(top5_unit)
    st.write(top5_unit_prob)
    # display in a table
    st.write("## 每個 unit 的 top 5 phonemes")
    top5_phn = dfcool.apply(
        lambda x: x.sort_values(ascending=False).head(5).index.tolist(),
        axis=0,
    )
    top5_phn_prob = dfcool.apply(
        lambda x: x.sort_values(ascending=False).head(5).values.tolist(),
        axis=0,
    )

    st.write(top5_phn)
    st.write(top5_phn_prob)

# Histogram of phoneme entropy
st.write(
    "Histogram of phoneme entropy"
)
st.write(
    h_each_phn.describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    
    )
)
st.write(
    h_each_phn.sort_values(ascending=False)
)
plot_barplot__series(
            h_each_phn,
            label='h_each_phn',
            color='red',
        )
plot_barplot__series(
            h_each_phn.sort_values(ascending=True),
            label='h_each_phn',
            color='red',
        )
fig = plt.figure(figsize=(10, 5))
sns.histplot(h_each_phn, 
            #  bins=int(analysis_obj[1]) // 3, 
             bins=25,
             kde=True,
             )
# xmin
plt.xlim(0, None)
st.pyplot(fig)
plt.close(fig)


### ref segL    hyp segL  p_xy shape      frm tot    frm diff    utt tot    utt miss
### ----------  ----------  ------------  ---------  ----------  ---------  ----------
###     4.8935      1.9733  (41, 100)      18088299          89      28539           0

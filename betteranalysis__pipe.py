import numpy as np
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, streamlit as st

# region --- Calculations ---
def entropy(p, axis=None):
    return -(p * np.log(p + 1e-8)).sum(axis=axis)

def calculate_pxy(
        p_xy: np.ndarray,  # shape = (num_y, num_x)
        xlabel="unit", ylabel="phn"):
    prob_y: np.ndarray = p_xy.sum(axis=1, keepdims=True)  # shape = (num_y, 1)
    prob_x: np.ndarray = p_xy.sum(axis=0, keepdims=True)  # shape = (1, num_x)
    cond_prob__x_given_y = p_xy / prob_y  # shape = (num_y, num_x)
    cond_prob__y_given_x = p_xy / prob_x  # shape = (num_y, num_x)

    y_entropy: float = entropy(prob_y)
    x_entropy: float = entropy(prob_x)
    entropy__x_given_y: np.ndarray = entropy(cond_prob__x_given_y, axis=1)  # shape = (num_y,)
    entropy__y_given_x: np.ndarray = entropy(cond_prob__y_given_x, axis=0)  # shape = (num_x,)

    max_each_y: np.ndarray = p_xy.max(axis=1)  # shape = (num_y,)
    max_each_x: np.ndarray = p_xy.max(axis=0)  # shape = (num_x,)
    x_purity: float = max_each_y.sum()
    y_purity: float = max_each_x.sum()
    argmax_each_y: np.ndarray = p_xy.argmax(axis=1)  # shape = (num_y,)
    argmax_each_x: np.ndarray = p_xy.argmax(axis=0)  # shape = (num_x,)

    if "Wei-ning's":  # assert np.isclose(mutual_info, mutual_info, atol=1e-4)
        pmi: np.ndarray = np.log(p_xy / np.matmul(prob_y, prob_x) + 1e-8)  # shape = (num_y, num_x)
        mutual_info: float = (p_xy * pmi).sum()
    else:  # alternative way to calculate mutual info
        joint_entropy: float = entropy(p_xy.flatten())
        mutual_info: float = (x_entropy + y_entropy - joint_entropy)

    return dict(values={
        f"{ylabel}_purity": y_purity,  # float
        f"{xlabel}_purity": x_purity,  # float
        f"{ylabel}_entropy": y_entropy,  # float
        f"{xlabel}_entropy": x_entropy,  # float
        f"mutual_info": mutual_info,  # float
        f"{ylabel}_NMI": mutual_info / y_entropy,  # float
        f"{xlabel}_NMI": mutual_info / x_entropy,  # float
    }, arrays={
        f"{ylabel}_prob": prob_y[:, 0],  # shape = (num_y,)
        f"{xlabel}_prob": prob_x[0, :],  # shape = (num_x,)
        f"entropy_{xlabel}_given_{ylabel}": entropy__x_given_y,  # shape = (num_y,)
        f"entropy_{ylabel}_given_{xlabel}": entropy__y_given_x,  # shape = (num_x,)
        f"max_{xlabel}_given_{ylabel}": max_each_y,  # shape = (num_y,)
        f"max_{ylabel}_given_{xlabel}": max_each_x,  # shape = (num_x,)
        f"{ylabel}_argmax": argmax_each_y,  # shape = (num_y,)
        f"{xlabel}_argmax": argmax_each_x,  # shape = (num_x,)
    }, matrices={
        f"{xlabel}_given_{ylabel}": cond_prob__x_given_y,  # shape = (num_y, num_x)
        f"{ylabel}_given_{xlabel}": cond_prob__y_given_x,  # shape = (num_y, num_x)
       # "pmi": pmi,  # shape = (num_y, num_x)
    })  
# endregion

# region --- Data loading ---
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
# endregion

# region --- Plotting ---
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
    # fig.set_size_inches(15, 10)
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
    **kwargs,
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
    
    # title


    st.pyplot(fig)
    plt.close(fig)
    return fig

# endregion

# region --- Data decoration ---
def decorate_data__copilot(
    p_xy: np.ndarray,
    xlabel: str = "unit",
    ylabel: str = "phn",
    x_groups = None,
    y_groups = None,
    horizontal_highlighted = False,
    vertical_highlighted = False,
    annotated = False,
):
    df = pd.DataFrame(p_xy, columns=xlabel, index=ylabel)
    df = df.div(df.sum(axis=1), axis=0)  # normalize
    df = df.sort_index(ascending=False)

    if x_groups is not None:
        df = df.groupby(np.arange(len(df)) // x_groups).sum()
    if y_groups is not None:
        df = df.groupby(np.arange(len(df)) // y_groups).sum()

    return df

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


def datalabel_decoration(p_xy__ndarray: np.ndarray, 
                         xlabel, 
                         ylabel,
                         domain_knowledge_df,
                         data,
                         ):
    kn_opt = st.selectbox(
        "knowledge type",
        [
            "haotang",
            "modified_alphabetical",
            "sonorant",
            "alphabetical",
        ],
    )
    if kn_opt == "alphabetical":
        domain_knowledge_df = "./knowledge_table.tsv"
    elif kn_opt == "sonorant":
        domain_knowledge_df = "./knowledge_table__phon.tsv"
    elif kn_opt == "modified_alphabetical":
        domain_knowledge_df = "./knowledge_table__jeff.tsv"
    elif kn_opt == "haotang":
        domain_knowledge_df = "./knowledge_table__hao.tsv"
    
    heatmap_cond = st.selectbox(
        "Heatmap conditional",
        [
            "phn given unit",
            "joint",
            "unit given phn",
        ],

    )

    knowledge_data = pd.read_csv(
        domain_knowledge_df, sep="\t"
    )

    lookup_phn_to_cls = dict(
        zip(knowledge_data['phn'], knowledge_data['cls'])
    )
    lookup_phn_to_pi = dict(
        zip(knowledge_data['phn'], knowledge_data['pi'])
    )

    PHONETIC_GROUPS = (
        knowledge_data['cls'].unique().tolist())
    PHONETIC_GROUP_NUMS = (
        knowledge_data.groupby('cls').count().loc[PHONETIC_GROUPS]['phn']
    ).tolist()
    # st.write(knowledge_data)


    p_xy__calculated = calculate_pxy(p_xy__ndarray)
    stats, arrays = p_xy__calculated["values"], p_xy__calculated["arrays"]
    mi, nmi_x, nmi_y = stats["mutual_info"], stats["unit_NMI"], stats["phn_NMI"]
    stats.pop("mutual_info")
    stats.pop("unit_NMI")
    stats.pop("phn_NMI")
    stats["PNMI"] = nmi_y
    if (3-2)*(0+1)+1: 
        st.dataframe(
        pd.DataFrame({"modeltype": modeltype, **stats}, index=[0]),
        hide_index=True,
    )
        st.dataframe(
        pd.DataFrame({
        'mi': mi,
        'UNMI': nmi_x,
        'ref_segL': data.get('refsegL').tolist(),
        'hyp_segL': data.get('hypsegL').tolist(),
        'p_xy_shape': [data.get('p_xy').shape],
        'frm_tot': data.get('tot'),
        'frm_diff': data.get('frmdiff'),
        'utt_tot': data.get('utt_tot'),
        'utt_miss': data.get('utt_miss'),
        }
        ),
        hide_index=True,
        )


    df = pd.DataFrame(
        {
            "joint": p_xy__ndarray, 
            "unit given phn": p_xy__calculated["matrices"]["unit_given_phn"],
            "phn given unit": p_xy__calculated["matrices"]["phn_given_unit"],
        }[heatmap_cond],
        columns=xlabel, index=ylabel)
    dfxy = pd.DataFrame(
        p_xy__ndarray,
        columns=xlabel, index=ylabel)
    df_unit_given_phn = pd.DataFrame(
        p_xy__calculated["matrices"]["unit_given_phn"],
        columns=xlabel, index=ylabel)
    df_phn_given_unit = pd.DataFrame(
        p_xy__calculated["matrices"]["phn_given_unit"],
        columns=xlabel, index=ylabel)
    
    stat_units = pd.DataFrame({
        "prob": arrays["unit_prob"],
        "entropy": arrays["entropy_phn_given_unit"],
        "max_prob": arrays["max_phn_given_unit"],
        "argmax": (ylabel)[arrays["unit_argmax"]],
    }, index=xlabel)
    stat_phns = pd.DataFrame({
        "prob": arrays["phn_prob"],
        "entropy": arrays["entropy_unit_given_phn"],
        "max_prob": arrays["max_unit_given_phn"],
        "argmax": np.array(xlabel)[arrays["phn_argmax"]],
    }, index=ylabel)
    stat_phns['cls'] = stat_phns.index.map(lookup_phn_to_cls)
    stat_phns['phnidx'] = stat_phns.index.map(lookup_phn_to_pi).astype(int)
    stat_units['argmax_cls'] = stat_units['argmax'].map(lookup_phn_to_cls)
    


    # plot_heatmap(df)

    # Step 1: Sort stat_phns by 'prob' and get the sorted indices
    SORTING_option = st.selectbox(
        "Sort by",
        [
         "by phonology", 
         "by probability", 
         "by entropy", 
         "alphabetical",
         ],
    )
    if SORTING_option == "by probability":
        order_phns = stat_phns.sort_values(by='prob', ascending=False).index
    elif SORTING_option == "by entropy":
        order_phns = stat_phns.sort_values(by='entropy', ascending=False).index
    elif SORTING_option == "alphabetical":
        order_phns = sorted(stat_phns.index.tolist())
    else:
        order_phns = stat_phns.sort_values(by='phnidx', ascending=True).index
    # order_phns = ['AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','sil','spn','T','TH','UH','UW','V','W','Y','Z','ZH']
    # order_phns = ['sil','spn','P','T','K','B','D','G','CH','JH','F','S','TH','SH','HH','V','Z','DH','ZH','M','N','NG','L','R','Y','W','IY','IH','UW','UH','EH','ER','AO','AE','AH','AA','EY','OW','OY','AY','AW']

    sorted_indices = dict({
        v: k for k, v in enumerate(order_phns)
    })
    if not not "by phonology":
        stzt = stat_units['argmax']

        stat_units['argmax_index_with_prob'] = (
            stzt.map(sorted_indices) + 1 - stat_units['max_prob']
        )
        order_units = stat_units.sort_values(by='argmax_index_with_prob', ascending=True).index
    else:
        order_units = stat_units.sort_values(by='prob', ascending=False).index

    if 0: st.write(stat_units)
    if 0: st.write(stat_units.loc[order_units])
    HYP_PHONETIC_GROUP_NUMS = stat_units.groupby('argmax_cls').count()
    # if some group is missing, add 0
    for item in PHONETIC_GROUPS:
        if item not in HYP_PHONETIC_GROUP_NUMS.index:
            HYP_PHONETIC_GROUP_NUMS.loc[item] = 0
    HYP_PHONETIC_GROUP_NUMS = HYP_PHONETIC_GROUP_NUMS.loc[PHONETIC_GROUPS]['argmax'].tolist()

    # if 1: st.write(HYP_PHONETIC_GROUP_NUMS)
    # st.write(order_phns)
    if 0: st.write(stat_phns)
    if 0: st.write(order_units)

    df_aligned_by_orders = df.loc[order_phns, order_units]
    # TODO: y_groups generator
    # find groups accoriding to stzt
    if 0: st.write(stzt)



    # st.write(df_aligned_by_orders)
    # st.write(stat_units)
    # stat_units_sorted = stat_units_sorted[sorted_indices]
    # st.write(stat_units_sorted)
    # Then, reorder df_sorted columns based on the sorted stat_units
    # df_sorted = df_sorted[stat_units_sorted['column_names']]
    # Optionally, plot the heatmap of the sorted df
    plot_heatmap(df_aligned_by_orders,
                    # x_groups=[20,12+1,1,2+2+1+1+1+1+1+1+1+1+1+1+1+1,0+1+1+1+1+1+1+1+1+1+1+1-3,0+1+6+1+3-1,20+4+2,99],
                    x_groups=HYP_PHONETIC_GROUP_NUMS if SORTING_option == "by phonology" else None,
                    y_groups=PHONETIC_GROUP_NUMS if SORTING_option == "by phonology" else None, 
                    horizontal_highlighted=True if SORTING_option == "by phonology" else False,
                    vmin=0,
                    vmax=0.025 if heatmap_cond == "joint" else 1,
                 )
    # plot_heatmap(df_sorted)
    # phn_prob = arrays["phn_prob"]
    # argsort_phn_prob = np.argsort(phn_prob)[::-1]

    # phn_by_argsort = np.array(ylabel)[argsort_phn_prob]
    # df = df.loc[phn_by_argsort, :]
    # plot_heatmap(df)

    # # align x with max
    # phn_argmax = arrays["phn_argmax"]
    # st.write(phn_argmax)
    # # phn_argmax = phn_argmax[argsort_phn_prob]


    # phn_prob = pd.Series(phn_prob, index=ylabel)
    # st.write(phn_prob)
    # make prob with their labels
    # sort by phn_prob
    # df = df.loc[phn_prob.index, :].sort_index(ascending=False)
    # plot_heatmap(df)

    # def align_sorter(df, sorter):
        # return df.loc[sorter, :].sort_index(ascending=False)
    # ss = (arrays["phn_prob"])
    # st.write(ss)
    # df = align_sorter(df, ss.index)
    # plot_heatmap(df)

    # st.markdown(f"## Statistics")
    plot_barplot__series(
        ser=stat_units.loc[order_units]['prob'],
        label="unit",
    )
    plot_barplot__series(
        ser=stat_phns.loc[order_phns]['prob'],
        label="phn",
        horizontal=True,
    )
        
    # histogram of phn entropy of each unit
    h_units = stat_units.loc[order_units]['entropy']
    # histo = np.histogram(h_units, bins=10)
    # st.write(histo)
    if 3-2:
        fig = plt.figure(figsize=(10, 5))
        sns.histplot(h_units, bins=20,
                    #  kde=True,
                     )
        from matplotlib.font_manager import FontProperties
        kai_font = FontProperties(
            fname='./fonts/BiauKai.ttf',
            weight='bold', size=12,
            # fontsize=12, fontdict={'family': 'DFKai-SB', 'weight': 'bold'})
        )
        plt.xlabel('音素熵', fontproperties=kai_font)
        plt.ylabel('數量', fontproperties=kai_font)
        plt.title('音素熵分布', fontproperties=kai_font)
        plt.xlim(0, None)
        st.pyplot(fig)
        plt.close(fig)
    # plot_barplot__series(
        # ser=,
        # label="unit",
    # )

    ##########################################3
    st.markdown("## Prob")
    prob_phns = pd.Series(arrays["phn_prob"], index=ylabel)
    prob_units = pd.Series(arrays["unit_prob"], index=xlabel)
    h_each_phn = pd.Series(arrays["entropy_unit_given_phn"], index=ylabel)
    h_each_unit = pd.Series(arrays["entropy_phn_given_unit"], index=xlabel)
    # st.write(h_each_unit)
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
    st.write("## 每個 unit 的 top 5 phonemes")

    t1=(
        df_aligned_by_orders.apply(
            lambda x: x.sort_values(ascending=False).head(5).index,
            axis=0,
        )
    ).T
    
    t2=(
        df_aligned_by_orders.apply(
            lambda x: x.sort_values(ascending=False).head(5).values,
            axis=0,
        )
    ).T
    t2 = (t2 * 100).round(2)
    t2 = t2.applymap(
        lambda cell:
        f"{cell:.2f} %"
    )
    # t3 = t1.iloc[0, :]
    # t3 = t1.iloc[0, :].apply(lookup_phn_to_cls.get)
    # t3 = t1.iloc[0:5, :].applymap(lookup_phn_to_cls.get)
    t3 = t1.applymap(lookup_phn_to_cls.get)

    zcombined_df = pd.DataFrame()

    for rankid in range(5):
        if rankid == 0:
            zcombined_df[f"pid" ] = t1.iloc[:, rankid].map(lookup_phn_to_pi)
        else:
            zcombined_df[str(rankid+1) ] = "|"
        zcombined_df[f"unit_{rankid + 1}"] = t1.iloc[:, rankid]
        zcombined_df[f"prob_{rankid + 1}"] = t2.iloc[:, rankid]
        zcombined_df[f"cls_{rankid + 1}" ] = t3.iloc[:, rankid]

    st.write(zcombined_df)
    st.write("## 每個 phoneme 的 top 5 units")


    tt1=(
        df_aligned_by_orders.T.apply(
            lambda x: x.sort_values(ascending=False).head(5).index,
            axis=0,
        )
    )  .T
    tt2=(
        df_aligned_by_orders.T.apply(
            lambda x: x.sort_values(ascending=False).head(5).values,
            axis=0,
        )
    )  .T
    tt2 = (tt2 * 100).round(2)
    tt2 = tt2.applymap(
        lambda cell:
        f"{cell:.2f} %"
    )
    tt3 = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    tt3.index = df_aligned_by_orders.index

    newcombined = pd.DataFrame()
    newcombined.index = df_aligned_by_orders.index
    newcombined['pid'] = df_aligned_by_orders.index.map(lookup_phn_to_pi)
    newcombined['cls'] = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    # newcombined[f"cls_{0}" ] = tt3
    # newcombined['a'+str(77) ] = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    for rankid in range(5):
        newcombined[str(rankid+1) ] = "|"
        newcombined[f"phn_{rankid + 1}" ] = tt1.iloc[ :, rankid]
        newcombined[f"prob_{rankid + 1}"] = tt2.iloc[ :, rankid]
    # newcombined[str(1) ] = ""
    # newcombined[str(66) ] = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    st.write(newcombined)


    visualize_phn = st.selectbox(
        "Select phoneme to visualize",
        order_phns,
    )
    visualize_unit = st.selectbox(
        "Select unit to visualize",
        order_units,
    )

    if 1:
        st.write(f"## Phoneme: {visualize_phn}")
        # p_xy__calculated["matrices"]["phn_given_unit"],
        # st.write(df_unit_given_phn)
        # st.write(df_unit_given_phn[order_units, order_phns])
        # st.write(df_unit_given_phn[order_units].loc[visualize_phn])
        # df_unit_given_phn = df_unit_given_phn.loc[:, order_phns]
        prob_this_phn = prob_phns.loc[visualize_phn]
        entropy_this_phn = stat_phns.loc[visualize_phn]['entropy']
        st.write(f"Probability: {prob_this_phn:.4f}")
        st.write(f"Entropy: {entropy_this_phn:.4f}")
        plot_barplot__series(
            df_unit_given_phn[order_units].loc[visualize_phn],
            # title="Phoneme given unit",
        )
    if 3-2:
        st.write(f"## Unit: {visualize_unit}")
        prob_this_unit = prob_units.loc[visualize_unit]
        entropy_this_unit = stat_units.loc[visualize_unit]['entropy']
        st.write(f"Probability: {prob_this_unit:.4f}")
        st.write(f"Entropy: {entropy_this_unit:.4f}")
        plot_barplot__series(
            df_phn_given_unit[visualize_unit].loc[order_phns])
        
    confusion_matrix = get_confusion_matrix(
        dfxy[order_units].loc[order_phns].values,
        HYP_PHONETIC_GROUP_NUMS,
        PHONETIC_GROUP_NUMS,
    )
    st.write(
        "Phone group accuracy:",
        np.round(
            (confusion_matrix * np.eye(*confusion_matrix.shape)).sum() * 100,
            2
        ),
        "%",
    )
    plot_heatmap(
        pd.DataFrame(
            confusion_matrix,
            index=PHONETIC_GROUPS,
            columns=PHONETIC_GROUPS,
        ),
        annotated=True,
    )


    confusion_matrix__cond = np.divide(
        confusion_matrix,
        confusion_matrix.sum(axis=1, keepdims=True),
    )
    plot_heatmap(
        pd.DataFrame(
            confusion_matrix__cond,
            index=PHONETIC_GROUPS,
            columns=PHONETIC_GROUPS,
        ),
        annotated=True,
    )

    confusion_matrix__cond0 = np.divide(
        confusion_matrix,
        confusion_matrix.sum(axis=0, keepdims=True),
    )
    plot_heatmap(
        pd.DataFrame(
            confusion_matrix__cond0,
            index=PHONETIC_GROUPS,
            columns=PHONETIC_GROUPS,
        ),
        annotated=True,
    )



    # return p_xy

# endregion

# region --- Main ---
modeltype, unit_num = parse_urlargs()
data = load_data(modeltype, unit_num)
@st.cache_resource
def loadtriphone(pathname):
    return np.load(pathname)
# data = loadtriphone(
#     '../mymeasure/tri_hubert_100.npz')
st.markdown(f"# {modeltype} --> {unit_num:3d} clusters")

# x as unit / hyp
# y as phone / ref
joint_prob: np.ndarray = data["p_xy"]  # shape = (num_phn, num_unit)
x_labels = [f"u{int(u):03d}" for u in data["hyp2lid"]]  # len = num_unit
# x_labels = data["hyp2lid"]  # len = num_unit
y_labels = data["ref2pid"]  # len = num_phn

datalabel_decoration(joint_prob, x_labels, y_labels, None, data)
# endregion

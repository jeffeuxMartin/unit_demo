import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.utils import entropy, calculate_pxy

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
        cmap="Greys",
        **options,
    )

    if df.shape[1] <= 1000:
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
    else:
        # hide xticks by replace with empty string
        pass
        axis.set_xticks([])
        axis.set_xticklabels([])
        # FIXME: still cannot 留白


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
    # add horizontal lines anyway first
    for idx in range(df.shape[0]):
        axis.axhline(idx, color='black', linewidth=0.2, linestyle="-")
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
            # hide xticks if too many
            if ser.shape[0] <= 100:
                sns.barplot(
                    x=ser.index,
                    y=ser.values,
                    ax=ax,
                    color=color,
                )
            else:
                sns.barplot(
                    x=ser.index,
                    y=ser.values,
                    ax=ax,
                    color=color,
                )
                ax.set_xticks([])
                ax.set_xticklabels([])
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
                         modeltype: str,
                         ):

    kn_opt = st.sidebar.selectbox(
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
    
    heatmap_cond = st.sidebar.selectbox(
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
    stats['rat'] = data.get('refsegL').tolist()/data.get('hypsegL').tolist()*(0.2007/(4.89354/4.99729))
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
    
    if 0: return       ####################
                       ####################
    # st.write(dfxy)   ####################
                       ####################
                       ####################
    # return           ####################
                       ####################
    st.write(stat_units)
    st.write(stat_phns)


    # plot_heatmap(df)

    # Step 1: Sort stat_phns by 'prob' and get the sorted indices
    SORTING_option = st.sidebar.selectbox(
        "Sort by",
        [
         "by probability", 
         "by phonology", 
         "by entropy", 
         "alphabetical",
         ],
    )

    with st.expander("Phonetic groups", True):

        # xxx_ratio
        silrat =df_phn_given_unit[stat_phns['cls'] == 'XXX'].sum(axis=0).sum()
        if 0:st.write(
            'sil_num: ',
            silrat,
        )
        st.write(
            'sil_ratio: ',
            silrat/df_phn_given_unit.values.sum(),
            # df_phn_given_unit.values.sum()
            # df_phn_given_unit.sum(axis=0).sum()
        )
        if 0:st.write(
            stat_phns
        )
        # group pxy by cls (phonetic group)
        if 0:st.write(
            stat_phns.groupby('cls').sum()
        )
        if 0:st.write(
            dfxy)
        dfxy_cls = dfxy.groupby(stat_phns['cls'], axis=0).sum()
        if 0:st.write(
            dfxy_cls
        )
        # get purity of dfxy_cls
        # st.write(dfxy_cls.max(axis=0))
        # st.write( dfxy_cls.sum(axis=0))
        # purity = dfxy_cls.max(axis=0) / dfxy_cls.sum(axis=0)
        pcls_purity = dfxy_cls.max(axis=0).sum()
        punit_purity = dfxy_cls.max(axis=1).sum()
        st.write(
            'purity of cls:',
            pcls_purity
        )
        st.write(
            'purity of unit in cls:',
            punit_purity
        )
        pass
        # 可以直接 concat 在 stat_phns 上，之後用表格顯示
        if 0:st.write(
            pd.DataFrame(
                {
                    "group": PHONETIC_GROUPS,
                    "num": PHONETIC_GROUP_NUMS,
                },
                index=np.arange(1, len(PHONETIC_GROUPS) + 1),
            )
        )
        # st.write(stat_phns)
        puirt_by_clsss = []
        for item in PHONETIC_GROUPS:
            # st.write(
                # "## Phonetic group: " + item
            # )
            xx = stat_phns.loc[stat_phns['cls'] == item]
            # purity of each phonetic group
            puirt_by_clsss.append((
                item,

                xx['max_prob'].sum() / xx['prob'].sum()
            )
            )
        # write in table
        dtailpur=pd.DataFrame(
                puirt_by_clsss,
                columns=['group', 'purity'],
            ).T
        # set header
        dtailpur.columns = dtailpur.iloc[0]
        dtailpur = dtailpur[1:]
        st.write(
            dtailpur)
        
        ##### %%%
        puirt_by_clsssReal = []






        for item in PHONETIC_GROUPS:
            # st.write(
                # "## Phonetic group: " + item
            # )
            v1 = dfxy.loc[dfxy.index.map(lookup_phn_to_cls) == item]
            subsum = v1.values.sum()
            
            rr = (
                v1.max(axis=0).sum() / subsum
            )
            puirt_by_clsssReal.append((
                item,

                rr
            )
            )
        # write in table
        dtailpurReal=pd.DataFrame(
                puirt_by_clsssReal,
                columns=['group', 'purity'],
            ).T
        # set header
        dtailpurReal.columns = dtailpurReal.iloc[0]
        dtailpurReal = dtailpurReal[1:]
        st.write(
            dtailpurReal)

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
    if "by phonology":
        reprPhnOfUnits = stat_units['argmax']


        stat_units['argmax_index_with_prob'] = (
            reprPhnOfUnits.map(sorted_indices) + 1 - stat_units['max_prob']
        )
        order_units = (
            stat_units.sort_values(by='argmax_index_with_prob', ascending=True).index)
    else:
        order_units = stat_units.sort_values(by='prob', ascending=False).index

    if 0: st.write(stat_units)
    if 0: st.write(stat_units.loc[order_units])
    HYP_PHONETIC_GROUP_NUMS = stat_units.groupby('argmax_cls').count()
    # if some group is missing, add 0
    for item in PHONETIC_GROUPS:
        if item not in HYP_PHONETIC_GROUP_NUMS.index:
            HYP_PHONETIC_GROUP_NUMS.loc[item] = 0
    HYP_PHONETIC_GROUP_NUMS = (
        HYP_PHONETIC_GROUP_NUMS.loc[PHONETIC_GROUPS]['argmax'].tolist())

    # if 1: st.write(HYP_PHONETIC_GROUP_NUMS)
    # st.write(order_phns)
    if 0: st.write(stat_phns)
    if 0: st.write(order_units)

    df_aligned_by_orders = df.loc[order_phns, order_units]
    # TODO: y_groups generator
    # find groups accoriding to stzt
    if 0: st.write(reprPhnOfUnits)
    # st.write(df_aligned_by_orders)


    # st.write(df_aligned_by_orders)
    # st.write(stat_units)
    # stat_units_sorted = stat_units_sorted[sorted_indices]
    # st.write(stat_units_sorted)
    # Then, reorder df_sorted columns based on the sorted stat_units
    # df_sorted = df_sorted[stat_units_sorted['column_names']]
    # Optionally, plot the heatmap of the sorted df
    plot_heatmap(df_aligned_by_orders,
                    # x_groups=[20,12+1,1,2+2+1+1+1+1+1+1+1+1+1+1+1+1,0+1+1+1+1+1+1+1+1+1+1+1-3,0+1+6+1+3-1,20+4+2,99],
                    x_groups=(
                        HYP_PHONETIC_GROUP_NUMS
                        if SORTING_option == "by phonology" else
                        None),
                    y_groups=(
                        PHONETIC_GROUP_NUMS
                        if SORTING_option == "by phonology" else
                        None),
                    horizontal_highlighted=(
                        True
                        if SORTING_option == "by phonology" else
                        False),
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
    def jeffcolormap(x):
        # print(x)
        return dict(
            XXX='gray',
            PLO='blue', #'cyan',
            AFF='#19e697',  # '#19e697',
            FRI='green', #'#4db368', # 'lime',
            NAS='#dd0', # '#c9c54b', #'yellow',
            APP='orange',
            VOW='red',
            DIP='#eb34a8',
        ).get(x, 'black')

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
    zout = pd.DataFrame()
    zcls = pd.DataFrame()

    for rankid in range(5):
        if rankid == 0:
            zcombined_df[f"pid" ] = t1.iloc[:, rankid].map(lookup_phn_to_pi)
        else:
            zcombined_df[str(rankid+1) ] = "|"
        zcombined_df[f"unit_{rankid + 1}"] = t1.iloc[:, rankid]
        zcombined_df[f"prob_{rankid + 1}"] = t2.iloc[:, rankid]
        zcombined_df[f"cls_{rankid + 1}" ] = t3.iloc[:, rankid]
        # zcombined_df['test'] = zcombined_df['unit_5'] + ' (' + zcombined_df['cls_5'] + ")"
        # zout[f'rank {rankid + 1}'] = zcombined_df[f'unit_{rankid + 1}'] + ' (' + zcombined_df[f'cls_{rankid + 1}'] + ')'
        zout[f'rank {rankid + 1}'] = zcombined_df[f'unit_{rankid + 1}']
        if rankid == 0:
            # zout[f'\u3000'] = ' '
            zout[f' '] = ' '
        # zout[f'cls {rankid + 1}'] = zcombined_df[f'cls_{rankid + 1}']
        zcls[f'rank {rankid + 1}'] = zcombined_df[f'cls_{rankid + 1}']
        if rankid == 0:
            # zcls[f'\u3000'] = ' '
            zcls[f' '] = ' '


    if 0:st.write(zcombined_df)
    # st.write(zout.T)
    # color zout by cls
    # zoutgood = zout.style.applymap(lambda x: f"color: {jeffcolormap(x)}").T
    # z__color = zcls.applymap(lambda x: f"color: {jeffcolormap(x)}")
    def color_based_on_phn(val, x):
        return f"color: {jeffcolormap(lookup_phn_to_cls.get(val))}"
    # zclsgood = zcls.style.applymap(lambda x: f"color: {jeffcolormap(x)}").T
    def apply_color(row, df2):
        return [
            color_based_on_phn(val, df2.loc[row.name, col])
            for col, val in row.items()

        ]
    zoutgood = zout.T.style.apply(apply_color, df2=zcls.T, axis=1)
    def apply_bold_line_between_rows(s):
        # 为第一行和第二行之间的单元格添加边框样式
        return ['border-bottom: 2px solid black' if i == 1 else '' for i in range(len(s))]
    zoutgood = zoutgood.apply(apply_bold_line_between_rows, axis=0)
    st.write(zoutgood)
    #############

    # st.write(zcombined_df.T)


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
    ttout = pd.DataFrame()
    newcombined.index = df_aligned_by_orders.index
    newcombined['pid'] = df_aligned_by_orders.index.map(lookup_phn_to_pi)
    newcombined['cls'] = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    # newcombined[f"cls_{0}" ] = tt3
    # newcombined['a'+str(77) ] = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    ttout['cls'] = newcombined['cls']
    for rankid in range(5):
        newcombined[str(rankid+1) ] = "|"
        newcombined[f"phn_{rankid + 1}" ] = tt1.iloc[ :, rankid]
        newcombined[f"prob_{rankid + 1}"] = tt2.iloc[ :, rankid]
        ttout[f'rank {rankid + 1}'] = newcombined[f'phn_{rankid + 1}']
    # newcombined[str(1) ] = ""
    # newcombined[str(66) ] = df_aligned_by_orders.index.map(lookup_phn_to_cls)
    # st.write(newcombined)
    IIP = lambda x: lookup_phn_to_pi.get(x)
    # st.write(
    ttout['myidx'] = (
        (ttout.index).map(IIP)
    )
    # )
    # ttout.sort_index(key=IIP, inplace=True)
    # ttout.sort_index(key=lambda x: lookup_phn_to_pi.get(x), inplace=True)
    # ttout
    ttout.sort_values(by='myidx', inplace=True)
    ttout.drop(columns='myidx', inplace=True)
    ttoutT = ttout.T

    # st.write(ttout)
    st.write(ttoutT)


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


    st.write(
        df_aligned_by_orders
    )



    # return p_xy

# endregion
# # fake table
# df = pd.DataFrame(p_xy__ndarray, columns=xlabel, index=ylabel)
# # text color
# st.dataframe(
#     df.style.applymap(
#         lambda cell: 'color: red' if cell > 0.0005 else 'color: black'
#     )
# )
# return
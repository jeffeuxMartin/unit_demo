import streamlit as st
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns


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

def entropy(p, axis=None):
    return -np.sum(p * np.log(p + 1e-8), axis=axis)

modeltype, unit_num = parse_urlargs()
data = load_data(modeltype, unit_num)
st.markdown(f"# {modeltype} - {unit_num:03d}")

# x as unit / hyp
# y as phone / ref
joint_prob: np.ndarray = data["p_xy"]  # shape = (num_phn, num_unit)
x_labels = data["hyp2lid"]  # len = num_unit
y_labels = data["ref2pid"]  # len = num_phn

prob_x = joint_prob.sum(axis=0, keepdims=True)  # shape = (1, num_unit)
prob_y = joint_prob.sum(axis=1, keepdims=True)  # shape = (num_phn, 1)
cond_prob_x_given_y = joint_prob / prob_y
cond_prob_y_given_x = joint_prob / prob_x

unit_entropy = entropy(prob_x)  # shape = (,)
phn_entropy = entropy(prob_y)  # shape = (,)
entropy_x_given_y = entropy(cond_prob_x_given_y, axis=1)  # shape = (num_phn,)
entropy_y_given_x = entropy(cond_prob_y_given_x, axis=0)  # shape = (num_unit,)

max_each_x = joint_prob.max(axis=0)
max_each_y = joint_prob.max(axis=1)
phn_purity = max_each_x.sum()
unit_purity = max_each_y.sum()

mutual_info = (
    joint_prob * np.log(
        joint_prob / np.matmul(prob_y, prob_x)
        + 1e-8)).sum()
# entropy_joint = entropy(joint_prob.flatten())
# mi2 = unit_entropy + phn_entropy - entropy_joint
# assert np.isclose(mutual_info, mi2, atol=1e-4)


st.dataframe(pd.DataFrame({
    "model_type": modeltype,
    'phn_purity': phn_purity,
    'unit_purity': unit_purity,
    'phn_entropy': phn_entropy,
    'unit_entropy': unit_entropy,
    'PNMI': mutual_info / phn_entropy,
}, index=[0]), hide_index=True)

st.dataframe(pd.DataFrame({
        'mi': mutual_info,
        'UNMI': mutual_info / unit_entropy,
        'ref_segL': data.get('refsegL').tolist(),
        'hyp_segL': data.get('hypsegL').tolist(),
        'p_xy_shape': [data.get('p_xy').shape],
        'frm_tot': data.get('tot'),
        'frm_diff': data.get('frmdiff'),
        'utt_tot': data.get('utt_tot'),
        'utt_miss': data.get('utt_miss'),
}), hide_index=True)

######################################################################
######################################################################
######################################################################

argmax_x = joint_prob.argmax(axis=0)
argmax_y = joint_prob.argmax(axis=1)
label_x = y_labels[argmax_x]
st.write("Most likely phone for each unit")
st.write(label_x)
# TODO: phn sort index


for xy_plot, plotname, statement in [
    (joint_prob, "Joint Probability",
    "All = 1"),
    (cond_prob_x_given_y, "Conditional Probability P(unit|phn)",
    "Each row = 1"),
    (cond_prob_y_given_x, "Conditional Probability P(phn|unit)",
    "Each column = 1"),
]:
    df = pd.DataFrame(xy_plot, index=y_labels, columns=x_labels),
    st.markdown(f"## {plotname}")
    st.markdown(f"##### {statement}")
    plot_heatmap(
        df
    )

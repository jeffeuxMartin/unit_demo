import streamlit as st
import numpy as np

# region --- Data loading ---
@st.cache_resource
def load_data(
    model_type: str = "hubert",
    cluster_num: int = 100,
    piece_vocab = None,
    triphone = False,
):
    if piece_vocab is None:  # single unit
        return np.load(
            "./analysis_data/"
            "singleunit{istriphone}/"
        f"{model_type}/"
        f"clu{cluster_num:03d}/"
            "train-clean-100.npz"
            .format(
                istriphone = (""  # normal phoneme
                    if not triphone else 
                    "_triphone"),
            )
        )
    else:
        return np.load(
            "./analysis_data/"
            "acousticpiece{istriphone}/"
        f"{model_type}/"
        f"clu{cluster_num:03d}/"
        f"apvocab{int(piece_vocab):05d}/"
            "train-clean-100.npz"
            .format(
                istriphone = (""
                    if not triphone else 
                    "_triphone"),
            )
        )


def parse_urlargs():
    urlmodeltype = st.query_params.get('modeltype', 'hubert')
    # inmymodeltype = st.text_input('modeltype', urlmodeltype)
    inmymodeltype = st.sidebar.selectbox('modeltype', [
        "hubert",
        'w2v2',
        'cpc',
        'logmel',
    ])
    if inmymodeltype != urlmodeltype:
        st.query_params['modeltype']=inmymodeltype
        urlmodeltype = inmymodeltype
    mymodeltype = urlmodeltype

    urlclu = st.query_params.get('clu', '100')
    # inmyclu = st.text_input('clu', urlclu)
    # inmyclu_new = st.selectbox('clu', ['100', '500', '1000', '2000', '5000', '10000'])
    # inmyclu = st.selectbox('clu', ['100', '500', '1000', '2000', '5000', '10000'], index=[0,1,2,3,4,5][int(urlclu)//100])
    inmyclu = st.sidebar.selectbox('clu', [str(item) for item in [
        50,
        100,
        200,
    ]])
    # select clu if urlclu selected
    if inmyclu != urlclu:
        st.query_params['clu']=inmyclu
        urlclu = inmyclu
    clu = int(urlclu)

    urlpiece = st.query_params.get('piece', "None")
    # inmypiece = st.text_input('piece', urlpiece)
    inmypiece = st.sidebar.selectbox('piece', [str(item) for item in [
        "None",
        *([100] if clu == 50 else []),
        500, 1000, 8000, 10000, 20000,
    ]])
    if inmypiece != urlpiece:
        st.query_params['piece']=inmypiece
        urlpiece = inmypiece
    mypiece = urlpiece if urlpiece != "None" else None

    if "triphone implemented":
        urltriphone = st.query_params.get('triphone', 'False')
        # inmytriphone = st.text_input('triphone', urltriphone)
        inmytriphone = st.sidebar.checkbox('triphone', urltriphone == 'True')
        if inmytriphone != urltriphone:
            st.query_params['triphone']=inmytriphone
            urltriphone = inmytriphone
        mytriphone = (urltriphone == 'True')
    else:
        mytriphone = False

    st.sidebar.markdown(" --- ")

    return mymodeltype, clu, mypiece, mytriphone
# endregion

def wrapit():
    # add a selector on the sidebar
    nouse = st.sidebar.selectbox(
        "Select a model",
        [
            "hubert",
            'w2v2',
            'cpc',
            'logmel',
        ],
    )


    st.markdown(" # No use selector above = " + nouse)
    # add a separator

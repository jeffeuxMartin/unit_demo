import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import streamlit as st

from src.utils import entropy, calculate_pxy
from src.plotting import datalabel_decoration
from src.utilother import load_data, parse_urlargs, wrapit


# region --- Main ---
if __name__ == "__main__":
    # wrapit()
    modeltype, unit_num, piece_vocab, is_triphone = parse_urlargs()
    data = load_data(modeltype, unit_num, piece_vocab, is_triphone)
    # st.markdown(f"# APiece hub50 ac1000")
    # st.markdown(f"# {modeltype} --> {unit_num:3d} clusters")
    if piece_vocab is not None:
        # TODO: triphone
        st.markdown(f"## {modeltype} --> {unit_num:3d} clus --> {piece_vocab} acpcs")
    else:
        st.markdown(f"## {modeltype} --> {unit_num:3d} clus")

    # x as unit / hyp
    # y as phone / ref
    joint_prob: np.ndarray = data["p_xy"]  # shape = (num_phn, num_unit)
    x_labels = [f"u{int(u):03d}" for u in data["hyp2lid"]]  # len = num_unit
    # x_labels = data["hyp2lid"]  # len = num_unit
    y_labels = data["ref2pid"]  # len = num_phn

    datalabel_decoration(joint_prob, x_labels, y_labels, None, data, modeltype)


# endregion

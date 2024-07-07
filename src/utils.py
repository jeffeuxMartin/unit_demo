import numpy as np
from typing import List, Tuple, Dict, Any

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

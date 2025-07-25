# utils/weights.py
import numpy as np
import torch

# 统一的 group 顺序，务必与 env.reward_groups 中保持一致
GROUPS = [
    "survival",
    "velocity_tracking",
    "base_pose",
    "smoothness",
    "joint_regularization",
    "contact",
]

_PRESETS = {
    #    survival  vel_track  base_pose  smooth  joint_reg  contact
    "A_Speed_Demon":       [0.05, 0.55, 0.05, 0.05, 0.05, 0.25],
    "B_Stable_Walker":     [0.10, 0.25, 0.30, 0.10, 0.10, 0.15],
    "C_Energy_Saver":      [0.10, 0.20, 0.10, 0.35, 0.15, 0.10],
    "D_Contact_Aware":     [0.10, 0.25, 0.10, 0.05, 0.15, 0.35],
    "E_All_Rounder":       [0.15, 0.25, 0.15, 0.15, 0.15, 0.15],
}

# --- 公开接口 ---------------------------------------------------------------

WEIGHTS_NP = {k: np.array(v, dtype=np.float32) for k, v in _PRESETS.items()}
WEIGHTS_TORCH = {k: torch.tensor(v, dtype=torch.float32) for k, v in _PRESETS.items()}

def get_weights(name: str, as_torch: bool = True):
    """name in {'A_Speed_Demon', ...}; returns 1×6 tensor/array"""
    if as_torch:
        return WEIGHTS_TORCH[name]
    return WEIGHTS_NP[name]


def get_normalized_single_weight(weights: np.ndarray, num_groups: int):
    """
    Convert any weights to standard weights.
    weights: np.ndarray, shape = (num_groups,)
    num_groups: int, number of groups
    """
    return weights / weights.sum() #* num_groups

def get_input_single_weight(weights: np.ndarray, num_groups: int, num_group_terms: np.ndarray, num_total_terms=26):
    """
    Convert any weights to standard weights.
    weights: np.ndarray, shape = (num_groups,)
    num_groups: int, number of groups
    """
    assert weights.sum() == 1, "normalized weights must sum to 1"

    weight_with_coeff = weights * num_groups / num_total_terms

    
    
    return input_weights

if __name__ == "__main__":
    a = [1,1,1,1,1,1]
    b = [1,5,3,6,3,8]
    weights = np.array(a, dtype=np.float32)
    print(get_normalized_single_weight(weights, 6))
# reward groups for MOPPO
import torch

REWARD_GROUP_INDEX = {
    "survival": 0,
    "velocity_tracking": 1,
    "base_pose": 2,
    "smoothness": 3,
    "joint_regularization": 4,
    "contact": 5,
}

REWARD_GROUPS = {
    "survival": [
        "survival",
    ],
    "velocity_tracking": [
        "tracking_lin_vel_x",
        "tracking_lin_vel_y",
        "tracking_ang_vel",
        "lin_vel_z",
        "ang_vel_xy",
    ],
    "base_pose": [
        "base_height",
        "orientation",
        "root_acc",
    ],
    "smoothness": [
        "torques",
        "torque_tiredness",
        "power",
        "dof_vel",
        "dof_acc",
        "action_rate",
    ],
    "joint_regularization": [
        "dof_pos_limits",
        "dof_vel_limits",
        "torque_limits",
    ],
    "contact": [
        "collision",
        "feet_slip",
        "feet_vel_z",
        "feet_yaw_diff",
        "feet_yaw_mean",
        "feet_roll",
        "feet_distance",
        "feet_swing",
    ]
}

def get_reward_to_group_index():
    """ tool for querying the index of groups given reward function names"""
    mapping = {}
    for group_name, reward_list in REWARD_GROUPS.items():
        group_idx = REWARD_GROUP_INDEX[group_name]
        for reward_name in reward_list:
            mapping[reward_name] = group_idx
    return mapping

REWARD_TO_GROUP = get_reward_to_group_index()

# reverse table: index --> name
GROUP_INDEX_TO_NAME = {v: k for k, v in REWARD_GROUP_INDEX.items()}

N_ALL = 26
GROUP_SIZES = torch.tensor([1, 5, 3, 6, 3, 8], dtype=torch.float32)
GROUP_SCALE = N_ALL / GROUP_SIZES

# 上调采样更集中 下调采样更分散
DIRICHLET_ALPHA_BASE = 0.1
DIRICHLET_ALPHA = DIRICHLET_ALPHA_BASE * GROUP_SIZES

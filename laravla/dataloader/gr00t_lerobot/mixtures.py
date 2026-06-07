"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
        # ("libero_90_no_noops_lerobot", 1.0, "libero_ur5"),
    ],

    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_local": [
        ("bridge_orig_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],
    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "agilex_cobot_magic_real4": [
        ("Agilex_Cobot_Magic_classify_object_fruit", 1.0, "agilex_cobot_magic"),
        ("Agilex_Cobot_Magic_pour_water_twice", 1.0, "agilex_cobot_magic"),
        ("Agilex_Cobot_Magic_stack_block_twice", 1.0, "agilex_cobot_magic"),
        ("Agilex_Cobot_Magic_storage_object_two", 1.0, "agilex_cobot_magic"),
    ],
    "agilex_cobot_magic_fruit": [
        ("Agilex_Split_Aloha_Storage_Fruits_Storing_fruits_1114", 1.0, "agilex_aloha"),
    ],
    "agilex_cobot_magic_pour": [
        ("Agilex_Split_Aloha_Pour_Water_Pour Water_1114", 1.0, "agilex_aloha"),
    ],
    "agilex_cobot_magic_stack": [
        ("Agilex_Split_Aloha_Stack_Blocks_Stack Blocks_1114", 1.0, "agilex_aloha"),
    ],
    "agilex_cobot_magic_storage": [
        ("Agilex_Split_Aloha_Storage_Objects_Storing Objects_1114", 1.0, "agilex_aloha"),
    ],

    # Agilex Aloha (single-task mixtures)
    "agilex_aloha_stack_bowl_1110": [
        ("Agilex_Split_Aloha_Stack_Bowl_Stacked_bowls_1110", 1.0, "agilex_aloha"),
    ],
    "agilex_aloha_storage_building_blocks_1109": [
        ("Agilex_Split_Aloha_Storage_Building_blocks_Storage_Building_Blocks_1109", 1.0, "agilex_aloha"),
    ],
    "agilex_aloha_storage_fruits_1114": [
        ("Agilex_Split_Aloha_Storage_Fruits_Storing_fruits_1114", 1.0, "agilex_aloha"),
    ],
    "agilex_aloha_storage_item_1124": [
        ("Agilex_Split_Aloha_Storage_ltem_Storage_items_1124", 1.0, "agilex_aloha"),
    ],

    # Agilex Aloha (4-task mixture)
    "agilex_aloha_real4": [
        ("Agilex_Split_Aloha_Stack_Bowl_Stacked_bowls_1110", 1.0, "agilex_aloha"),
        ("Agilex_Split_Aloha_Storage_Building_blocks_Storage_Building_Blocks_1109", 1.0, "agilex_aloha"),
        ("Agilex_Split_Aloha_Storage_Fruits_Storing_fruits_1114", 1.0, "agilex_aloha"),
        ("Agilex_Split_Aloha_Storage_ltem_Storage_items_1124", 1.0, "agilex_aloha"),
    ],

    # Galaxea R1 Lite (single dataset, task_id=1105)
    "galaxea_r1_lite_storage_1105_single": [
        (
            "Galaxea_R1_Lite_Storage_Building blocks_Storage Building Blocks_1105",
            1.0,
            "galaxea_r1_lite_storage_1105",
        ),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


}

import os
import pytest

from compute_lmc import visualize_conf_matrix


def test_visualize_lmc():
    json_root_dir = 'lmc_data'
    dataset_name = 'ruin_names'
    visualize_dir = 'visualize'
    json_path = os.path.join(json_root_dir, dataset_name, 'acc_list_seed1,2,3,4,5.json')

    visualize_conf_matrix(
        json_path=json_path,
        save_dir=visualize_dir,
        dataset_name=dataset_name,
        scale=10,
    )
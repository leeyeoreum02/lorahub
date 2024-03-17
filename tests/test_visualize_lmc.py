import os
import pytest

from compute_lmc import visualize_lmc


def test_visualize_lmc():
    json_root_dir = 'lmc_data'
    dataset_name = 'dyck_languages'
    visualize_dir = 'visualize'
    json_path = os.path.join(json_root_dir, dataset_name, 'loss_list_seed1,2,3,4,5.json')

    visualize_lmc(
        json_path=json_path,
        save_dir=visualize_dir,
        dataset_name=dataset_name,
        reverse=False,
    )

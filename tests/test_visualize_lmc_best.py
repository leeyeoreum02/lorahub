import os
import pytest

from compute_lmc import visualize_lmc_best


def test_visualize_lmc_best():
    json_root_dir = 'best_lmc_data'
    dataset_name = 'boolean_expressions'
    visualize_dir = 'visualize'
    json_path = os.path.join(json_root_dir, dataset_name, 'acc_list_best20.json')

    visualize_lmc_best(
        json_path=json_path,
        save_dir=visualize_dir,
        dataset_name=dataset_name,
        reverse=False,
    )

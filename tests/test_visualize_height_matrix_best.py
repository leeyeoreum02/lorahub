import os
import pytest

from compute_lmc import visualize_conf_matrix_best


def test_visualize_lmc():
    json_root_dir = 'best_lmc_data'
    perf_json_dir = 'acc_every_module'
    dataset_name = 'boolean_expressions'
    visualize_dir = 'visualize'
    json_path = os.path.join(json_root_dir, dataset_name, 'acc_list_best20.json')

    visualize_conf_matrix_best(
        json_path=json_path,
        save_dir=visualize_dir,
        perf_json_dir=perf_json_dir,
        dataset_name=dataset_name,
    )

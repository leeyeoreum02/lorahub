import os
import pytest

from compute_perf import visualize_perf_per_module


def test_visualize_lmc():
    json_root_dir = 'acc_per_module'
    dataset_name = 'ruin_names'
    visualize_dir = 'visualize'
    json_path = os.path.join(json_root_dir, dataset_name, 'acc_per_modules_seed5.json')

    visualize_perf_per_module(
        json_path=json_path,
        save_dir=visualize_dir,
        dataset_name=dataset_name,
    )

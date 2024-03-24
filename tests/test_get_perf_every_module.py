import pytest

from compute_perf import get_perf_every_module


def test_visualize_lmc():
    test_data_dir = 'data_bbh'
    json_dir = 'acc_every_module'
    batch_size = 32
    dataset_indices = [4, 6, 17]

    get_perf_every_module(
        folder=test_data_dir,
        batch_size=batch_size,
        save_dir=json_dir,
        # dataset_indices=dataset_indices,
        dataset_indices=list(range(13, 27)),
    )

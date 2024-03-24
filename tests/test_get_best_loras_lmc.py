import pytest

from compute_lmc import get_best_loras_lmc


def test_get_loras_lmc():
    folder = 'data_bbh'
    perf_json_dir = 'acc_every_module'
    batch_size = 32
    save_dir = 'best_lmc_data'
    err_type = 'acc'
    dataset_indices = [0]

    get_best_loras_lmc(
        folder=folder,
        perf_json_dir=perf_json_dir,
        batch_size=batch_size,
        save_dir=save_dir,
        err_type=err_type,
        # dataset_indices=dataset_indices,
    )

import pytest

from lorahub_methods import lorahub_with_best_lmc


def test_lorahub_with_best_lmc():
    dataset_dir = 'data_bbh'
    lmc_json_dir = 'best_lmc_data'
    save_dir = 'acc_lorahub_with_best_lmc'
    dataset_indices = [0, 6]

    lorahub_with_best_lmc(
        dataset_dir=dataset_dir,
        lmc_json_dir=lmc_json_dir,
        save_dir=save_dir,
        dataset_indices=dataset_indices,
    )

import pytest

from lorahub_methods import mean_lora_weights_with_best_lmc


def test_mean_lora_weights_with_best_lmc():
    dataset_dir = 'data_bbh'
    lmc_json_dir = 'best_lmc_data'
    save_dir = 'acc_mean_lora_weights_with_best_lmc'
    dataset_indices = [0, 6]

    mean_lora_weights_with_best_lmc(
        dataset_dir=dataset_dir,
        lmc_json_dir=lmc_json_dir,
        save_dir=save_dir,
        dataset_indices=dataset_indices,
    )

import pytest

from lorahub_methods import model_soups_with_just_best


def test_model_soups_with_just_best():
    dataset_dir = 'data_bbh'
    acc_json_dir = 'acc_every_module'
    save_dir = 'acc_model_soups_with_just_best'
    dataset_indices = list(range(13))

    model_soups_with_just_best(
        dataset_dir=dataset_dir,
        acc_json_dir=acc_json_dir,
        save_dir=save_dir,
        dataset_indices=dataset_indices,
    )

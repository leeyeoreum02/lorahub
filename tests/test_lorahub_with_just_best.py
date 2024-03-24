import pytest

from lorahub_methods import lorahub_with_just_best, default_l2_regularization


def test_lorahub_with_just_best():
    dataset_dir = 'data_bbh'
    acc_json_dir = 'acc_every_module'
    save_dir = 'acc_lorahub_with_just_best_l2_nolim'
    dataset_indices = list(range(13))

    sampling_seed_list=[24, 42, 100]
    for seed in sampling_seed_list:
        lorahub_with_just_best(
            dataset_dir=dataset_dir,
            acc_json_dir=acc_json_dir,
            save_dir=save_dir,
            # dataset_indices=dataset_indices,
            sampling_seed=seed,
            get_regular=default_l2_regularization,
            weight_range_limit=False,
        )

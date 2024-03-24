import pytest

from lorahub_methods import lorahub_greedy_soups, default_l2_regularization


def test_lorahub_with_just_best():
    dataset_dir = 'data_bbh'
    acc_json_dir = 'acc_every_module'
    save_dir = 'acc_lorahub_greedysoups'
    
    dataset_indices = [
        0, 1, 2, 5, 6, 8, 9, 10, 14, 15,
        16, 17, 18, 19, 20, 21, 25, 7, 11, 
        13, 3, 4, 12, 22, 23, 24, 26,    
    ]

    sampling_seed_list=[24, 42, 100]
    for seed in sampling_seed_list:
        lorahub_greedy_soups(
            dataset_dir=dataset_dir,
            acc_json_dir=acc_json_dir,
            save_dir=save_dir,
            dataset_indices=dataset_indices,
            sampling_seed=seed,
            get_regular=default_l2_regularization,
            # weight_range_limit=False,
        )

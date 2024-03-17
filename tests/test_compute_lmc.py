import pytest

from compute_lmc import get_loras_lmc


def test_get_loras_lmc(request):
    folder = request.config.getoption('--folder')
    batch_size = request.config.getoption('--batch-size')
    save_dir = request.config.getoption('--save-dir')
    err_type = request.config.getoption('--err-type')
    dataset_indices = request.config.getoption('--dataset-indices')
    n_seed= request.config.getoption('--n-seed')

    get_loras_lmc(
        folder=folder,
        batch_size=batch_size,
        save_dir=save_dir,
        err_type=err_type,
        dataset_indices=dataset_indices,
        n_seed=n_seed,
    )

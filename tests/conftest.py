def pytest_addoption(parser):
    parser.addoption('--save-dir', action='store', type=str)
    parser.addoption('--err-type', action='store', type=str)
    parser.addoption('--folder', action='store', type=str, default='data_bbh')
    parser.addoption('--batch-size', action='store', type=int, default=10)
    parser.addoption('--dataset-indices', action='store', nargs='+', type=int, default=None)
    parser.addoption('--n-seed', action='store', type=int, default=5)

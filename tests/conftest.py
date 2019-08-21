import pytest
import pandas as pd

TEST_DATA = 'tests/test_data'

@pytest.fixture(scope='function')
def fixture_test_data():
    return pd.read_pickle(TEST_DATA)

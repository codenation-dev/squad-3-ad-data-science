import pandas as pd

from squad_3_ad_data_science.feature_engineering import pipeline_data, select_features
from squad_3_ad_data_science import config

def test_data(fixture_test_data):
    assert type(fixture_test_data) == pd.DataFrame

def test_pipeline_data(fixture_test_data):
    pipelined_data, sp_labels = pipeline_data(fixture_test_data.copy())

    # Grants that all ids are comming back
    assert (set(pipelined_data.index) - set(fixture_test_data['id'])) == set()
    # Grant that ids column was removed
    assert 'id' not in pipelined_data.columns
    # Grant that sp_labels was correctly created
    assert all([
        'sp_' + col in sp_labels.columns for col in config.SPECIAL_LABELS
    ])
    # Grant no NaNs on pipelined data
    assert pipelined_data.isna().sum().sum() == 0

def test_feature_selection(fixture_test_data):
    pipelined_data, sp_labels = pipeline_data(fixture_test_data)
    feature_selected = select_features(pipelined_data)

    # Grants that all ids are comming back
    assert (set(feature_selected.index) - set(pipelined_data.index)) == set()
import pandas as pd

from squad_3_ad_data_science.feature_engineering import (pipeline_data,
                                                         select_features)
from squad_3_ad_data_science.model_training import create_mini_batch_kmeans
from squad_3_ad_data_science import config


def test_model_creation(fixture_test_data):
    pipelined_data, sp_labels = pipeline_data(fixture_test_data)
    feature_selected = select_features(pipelined_data)

    ret = create_mini_batch_kmeans(feature_selected.copy())

    # Grant flag is working and returning labels for all rows
    assert type(ret) != tuple
    ret = create_mini_batch_kmeans(feature_selected, return_labels=True)
    assert type(ret) == tuple
    assert len(ret[1]) == len(feature_selected.index)

    
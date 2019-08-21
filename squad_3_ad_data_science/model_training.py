import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from squad_3_ad_data_science import config


def create_mini_batch_kmeans(input_data: pd.DataFrame, return_labels=False):
    '''
        @brief Create MiniBatchKMeans trained with data and config.N_CLUSTERS

        @param input_data: input dataframe

        @return MiniBatchKmeans trained object. If `return_labels`, return
                labels for training data
    '''

    model = MiniBatchKMeans(n_clusters=config.N_CLUSTERS, batch_size=500)
    labels = model.fit_predict(input_data)

    if return_labels:
        return (model, labels)

    else:
        return model

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from squad_3_ad_data_science import config


def make_recomendation(market_list: pd.DataFrame, user_ids: list,
                       k, use_clusters=True):
    '''
        @brief Take market and ids that are client of user and return
        a number of recomendations K
        Ids present in user_ids will be removed from market_list


        @param market_list: Data of entire market.
        @param user_ids: clients of user
        @param k: number of recomendations

        @return list of ids from market_list, ordered by score
    '''

    user_rows = market_list.loc[user_ids]
    market_no_user = market_list.drop(labels=user_ids,
                                      axis=0,
                                      errors='ignore')

    # If use_clusters, remove ocurrences that does'nt share a cluster
    # with users_ids
    if use_clusters:
        # Secure that labels are present on data
        if 'label' not in market_list.columns:
            logger.error('Use clusters was set as `True`,' +
                         ' but column `label` do not exist')
            exit(1)

        user_labels = user_rows['label'].value_counts().index

        market_no_user = market_no_user[[d in user_labels
                                         for d in market_no_user['label']]]

        sim = cosine_similarity(market_no_user, user_rows)
        scores = np.amax(sim, axis=1)

        market_no_user['scores'] = scores

        market_no_user.sort_values(by=['scores'],
                                   inplace=True,
                                   ascending=False)

        return list(market_no_user.index)


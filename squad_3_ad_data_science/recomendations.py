import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from squad_3_ad_data_science import config


def make_recomendation(market_list: pd.DataFrame, user_ids: list,
                       use_clusters=False, use_sp_labels=True):
    '''
        @brief Take market and ids that are client of user and return
        market ordered by score
        Ids present in user_ids will be removed from market_list


        @param market_list: Data of entire market.
        @param user_ids: clients of user

        @return list of ids from market_list, ordered by score
    '''

    user_rows = market_list.loc[user_ids]
    market_no_user = market_list.drop(labels=user_ids)

    # If use_clusters, remove ocurrences that does'nt share a cluster
    # with users_ids
    if use_clusters:
        logger.info(f'Using clusters to eliminate rows')
        # Secure that labels are present on data
        if config.CLUSTER_LABEL not in market_list.columns:
            logger.error('Use clusters was set as `True`,' +
                         f' but column `{config.CLUSTER_LABEL}` do not exist')
            exit(1)

        user_labels = user_rows[config.CLUSTER_LABEL].value_counts().index
        # Remove from market_no_user all rows that don't share a cluster label
        # with user
        market_no_user = market_no_user[[d in user_labels
                                         for d in
                                         market_no_user[config.CLUSTER_LABEL]]]

    # Drop cluster label
    market_no_user.drop(config.CLUSTER_LABEL, inplace=True, axis=1)
    user_rows.drop(config.CLUSTER_LABEL, inplace=True, axis=1)
    if use_sp_labels:
        # Remove rows that don't share special_cols with list of
        # special cols values of user
        logger.info(f'Using special labels to eliminate' +
                    f' rows: {config.SPECIAL_LABELS}')

        for col in config.SPECIAL_LABELS:
            sp = 'sp_' + col
            user_sp_label = list(user_rows[sp].unique())
            selection = market_no_user[sp].isin(user_sp_label)
            market_no_user = market_no_user.loc[selection]

    # Drop special labels
    for col in config.SPECIAL_LABELS:
        user_rows.drop('sp_'+col, axis=1)
        market_no_user.drop('sp_'+col, axis=1)

    sim = cosine_similarity(market_no_user, user_rows)
    # scores = np.amax(sim, axis=1)
    scores = sim.sum(axis=1)

    market_no_user['scores'] = scores

    market_no_user.sort_values(by=['scores'],
                               inplace=True,
                               ascending=False)

    return list(market_no_user.index)

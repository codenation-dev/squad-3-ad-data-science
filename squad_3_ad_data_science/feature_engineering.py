import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from squad_3_ad_data_science import config


def pipeline_data(input_data: pd.DataFrame):
    '''
        @brief receive the input dataframe and apply treatment of
        NaN values and select most important features

        @param input_data: input dataframe

        @return dataframe with the feature selection and NaN treatment
    '''
    # Remove id, set as index
    try:
        input_data.set_index('id', inplace=True)

    except KeyError:
        logger.error('`id` col is not present. Check your train data.')
        exit(1)

    # Remove variables configured on config.TO_REMOVE
    input_data.drop(axis=1, labels=config.TO_REMOVE, inplace=True)

    # Remove variables with NaN count > NAN_THRESH
    removed_columns = []
    for col in input_data.iloc[:]:
        nan_pctg = input_data[col].isna().sum()/input_data[col].shape[0]
        if nan_pctg > config.NAN_THRESH:
            removed_columns.append(col)

    input_data.drop(removed_columns, inplace=True, axis=1)

    # Fix True/False objects
    try:
        for col in config.TO_FIX_OBJ2BOOL:
            input_data[col] = input_data[col].astype('bool')

    except KeyError:
        logger.error(f'Col {col} is not present. Check your train data or ' +
                     'config.TO_FIX_OBJ2BOOL.')
        exit(1)

    # Impute as configured on config.NAN_FIXES
    try:
        input_data = impute_as_config(input_data)

    except ConfigMissException:
        logger.error(f'Error ocurred while imputing data')
        raise

    # Grant that all NaNs are solved.
    if input_data.isna().sum().sum() != 0:
        cols = input_data.columns[input_data.isna().any()].tolist()

        logger.error(f'Fail to treat NaNs. Cols {cols} still have NaN' +
                     f'values. Total: {len(cols)} columns.')
        exit(1)

    # Make one hot encoding
    input_data = pd.get_dummies(input_data,
                                sparse=True,
                                drop_first=True)

    # Scale data
    scaler = StandardScaler()
    input_data[:] = scaler.fit_transform(input_data)

    return input_data


def impute_as_config(input_data: pd.DataFrame):
    '''
        @brief use values on config.NAN_FIXES to treat features

        @param input_data: input dataframe

        @return input_dataframe with columns filled as configured
    '''
    for col, info in config.NAN_FIXES.items():
        try:
            input_data[col]
        except KeyError:
            raise ConfigMissException(f'Column {col} is not present on ' +
                                      'dataframe')

        try:
            if info['method'] == 'const':
                input_data[col] = input_data[col].fillna(value=info['value'])

            elif info['method'] == 'median':
                median = input_data[col].quantile()
                input_data[col] = input_data[col].fillna(value=median)

            elif info['method'] == 'mean':
                mean = input_data[col].quantile()
                input_data[col] = input_data[col].fillna(value=mean)

            elif info['method'] == 'mode':
                mode = input_data[col].quantile()
                input_data[col] = input_data[col].fillna(value=mode)

            else:
                raise KeyError

        except KeyError:
            raise ConfigMissException(f'config.NAN_FIXES is with some value ' +
                                      f'error on col {col} with info = {info}')

    return input_data


class ConfigMissException(Exception):
    '''
        @brief Exception for fails on configuration file
    '''
    pass


def select_features(input_data: pd.DataFrame):
    '''
        @brief Select features to use based on describing
               config.EXPLAINED_VARIANCE_RATIO of data

        @param input_data: input dataframe

        @return input_dataframe with only columns
    '''

    # Create PCA
    pca = PCA()
    pca.fit_transform(input_data)

    ratio = pca.explained_variance_ratio_

    counter = 0
    value = 0

    # Check how many variables we need to keep if we want to represent
    # 60% os variance
    while value <= config.EXPLAINED_VARIANCE_RATIO:
        value = np.cumsum(ratio)[counter]
        counter += 1

    num_components = int(counter)
    pca = PCA(n_components=num_components)
    input_data = pd.DataFrame(data=pca.fit_transform(input_data),
                              index=input_data.index)

    return input_data

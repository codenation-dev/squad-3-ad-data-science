import pickle  # nosec
import json
from os.path import isfile, join, basename, splitext
from datetime import datetime

import pandas as pd
import fire
from loguru import logger
from sklearn.model_selection import train_test_split

from squad_3_ad_data_science import config  # noqa
from squad_3_ad_data_science.model_training import create_mini_batch_kmeans
from squad_3_ad_data_science.feature_engineering import (pipeline_data,
                                                         select_features)
from squad_3_ad_data_science.recomendations import make_recomendation
from squad_3_ad_data_science.validation import custom_apk


def features(**kwargs):
    """Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.

    NOTE
    ----
    config.data_path: workspace/data

    You should use workspace/data to put data to working on.  Let's say
    you have workspace/data/iris.csv, which you downloaded from:
    https://archive.ics.uci.edu/ml/datasets/iris. You will generate
    the following:

    + workspace/data/test.csv
    + workspace/data/train.csv
    + workspace/data/validation.csv
    + other files

    With these files you can train your model!
    """
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")

    # As we do not count on obtaining new dataset dinamically, this function
    # just checks if the expected datasets are present on the right folder
    if not isfile(config.train_dataset):
        logger.error(f'Train dataset `{config.train_dataset}` not found.' +
                     'Model cannot be trained.')

        exit(1)

    else:
        logger.info(f'Train dataset `{config.train_dataset}` found!')

    for test_ds in config.test_datasets:
        if not isfile(test_ds):
            logger.error(f'Test dataset `{test_ds}` not found.' +
                         'Model may not be tested. Fix or remove from tests.')

        else:
            logger.info(f'Test dataset `{test_ds}` found!')

    logger.info('Loading data')
    pure_data = pd.read_csv(config.train_dataset, index_col=0)

    old_shape = pure_data.shape
    logger.info('Working on features')
    try:
        data, special_labels_df = pipeline_data(pure_data)
    except Exception as e:
        logger.error(f'Error during pipeline. Exception text: {e}')
        exit(1)

    logger.info('Selecting variables with PCA')

    data = select_features(data)
    # Less 1, disconsidering cluster
    logger.info(f'Selected {len(data.columns)-1} features for' +
                f' a {config.EXPLAINED_VARIANCE_RATIO*100}% variance' +
                f' representation')
    logger.info(f'Concatenating special labels')

    data = data.join(special_labels_df)

    logger.info(f'Data original shape: {old_shape}')
    logger.info(f'Data training shape: {data.shape}')

    # Save dataset as pickle file
    data_path = join(config.data_path, config.TRAIN_DATA_PKL)
    data.to_pickle(data_path)
    logger.info(f'Data saved as `{data_path}`')


def train(**kwargs):
    """Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING YOUR MODEL!")

    # Load data and create model
    data = pd.read_pickle(join(config.data_path, config.TRAIN_DATA_PKL))
    model, labels = create_mini_batch_kmeans(data, return_labels=True)

    # Save data with cluster label
    data[config.CLUSTER_LABEL] = labels
    labeld_data_pkl_path = join(config.data_path,
                                config.TRAIN_DATA_PKL_LABELED)
    data.to_pickle(labeld_data_pkl_path)

    logger.info(f'Labeled data saved as `{labeld_data_pkl_path}`')

    # Save model
    model_path = join(config.models_path, config.MBK_PICKLE)
    with open(model_path, 'wb') as f:
        pickle.dump(model, file=f)

    logger.info(f'Model saved as `{model_path}`')


def metadata(**kwargs):
    """Generate metadata for model governance using testing!

    Kwargs:
        update: Write new registers for equal names. Default: True

    NOTE
    ----
    workspace_path: config.workspace_path

    In this section you should save your performance model,
    like metrics, maybe confusion matrix, source of the data,
    the date of training and other useful stuff.

    You can save like as workspace/performance.json:

    {
       'name': 'My Super Nifty Model',
       'metrics': {
           'accuracy': 0.99,
           'f1': 0.99,
           'recall': 0.99,
        },
       'source': 'https://archive.ics.uci.edu/ml/datasets/iris'
    }
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")

    # Kwargs processing
    if 'update' in kwargs.keys():
        update = kwargs['update']
        logger.info(f'Update flag found.')

    else:
        update = True

    # Load saved labeled data
    labeld_data_pkl_path = join(config.data_path,
                                config.TRAIN_DATA_PKL_LABELED)
    data = pd.read_pickle(labeld_data_pkl_path)

    # Try to load previous test performance file, to add new tests
    if isfile(config.performance_metadata_path):
        logger.info(f'File `{config.performance_metadata_path}` ' +
                    'found. Reading...')
        with open(config.performance_metadata_path) as jf:
            perf = json.load(jf)

    else:
        logger.info(f'No previous perfomance file found. Creatings new...')
        perf = dict()

    # Run over each test file, loading its ids and running the prediction
    for f in config.test_datasets:
        if not isfile(f):
            logger.error(f'Test file `{f}` not find. Ignoring...`')
            continue

        logger.info(f'Processing file `{f}`...')

        # Take from filepath the filename without .csv
        base = basename(f)
        name = splitext(base)[0]

        user_data = pd.read_csv(f, index_col=0)
        user_ids = user_data['id'].tolist()

        train, test = train_test_split(user_ids,
                                       test_size=config.TEST_SIZE,
                                       random_state=42)

        apk_values = dict()
        for k in config.APK_VALUES:
            ordered_recs = make_recomendation(data, train)

            score = custom_apk(market_list=ordered_recs,
                               test_list=test,
                               k=k)

            apk_values[f'apk@{k}'] = score

        # rename name to something to avoid overwrite older
        if not update and name in perf.keys():
            c = 1
            while name + f'_{c}' in perf.keys():
                c += 1

            name = name + f'_{c}'

        perf[name] = {
            'last_update': str(datetime.now()),
            'from_file': f,
            'test_size': config.TEST_SIZE,
            'scores': apk_values,
        }

    with open(config.performance_metadata_path, 'w') as f:
        json.dump(perf, f, indent=4)

    logger.info(f'Metadata generation finished. Results ' +
                f'available in `{config.performance_metadata_path}`')


def predict(input_data, **kwargs):
    """Predict: load the trained model and score input_data

        @kwargs:
            --k <value>: number of leads, default 10
            --use_clusters <True/False>: To use clusters,
                                         default False
            --use_sp_labels <True/False>: To use special labels from config,
                                          default True
            --return <True/False>: To return recomendations. If False, func
                                    will print then. Default: False

    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """
    print("==> PREDICT DATASET {}".format(input_data))
    print("Args: {}".format(kwargs))

    # Arguments reading
    if 'k' in kwargs.keys():
        k = kwargs['k']
        logger.info(f'K argument found. K={k}')

    else:
        logger.info(f'K=10')
        k = 10

    if 'use_clusters' in kwargs.keys():
        use_clusters = kwargs['use_clusters']
    else:
        use_clusters = False

    if 'use_sp_labels' in kwargs.keys():
        use_sp_labels = kwargs['use_sp_labels']
    else:
        use_sp_labels = True

    if 'return' in kwargs.keys():
        return_recs = kwargs['return']

    else:
        return_recs = False

    # Grant that input_data exists
    if not isfile(input_data):
        logger.error(f'File `{input_data}` not find!')
        exit(1)

    data = pd.read_csv(input_data)

    try:
        ids = data['id']

    except KeyError:
        logger.error(f'Missing id column on input data')
        exit(1)

    # Load trained data
    labeld_data_pkl_path = join(config.data_path,
                                config.TRAIN_DATA_PKL_LABELED)
    if not isfile(labeld_data_pkl_path):
        logger.error(f'Trained and labeled data do not exist! ' +
                     'Train model first.')
        exit(1)

    train_data = pd.read_pickle(labeld_data_pkl_path)

    ordered_ids = make_recomendation(train_data, ids,
                                     use_clusters=use_clusters,
                                     use_sp_labels=use_sp_labels)

    recomendations = ordered_ids[:k]
    if not return_recs:
        print('-- 0 is the most recomended lead. --')
        print(f'| rank  |  id')
        for i, r in enumerate(recomendations):
            print(f'|   {i}   | {r}')

    else:
        return recomendations


# Run all pipeline sequentially
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running squad_3_ad_data_science by ")
    features(**kwargs)  # generate dataset for training
    train(**kwargs)     # training model and save to filesystem
    metadata(**kwargs)  # performance report


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()

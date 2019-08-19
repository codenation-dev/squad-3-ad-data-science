from os import path

import squad_3_ad_data_science

# Datasets to use as test
TRAIN_DATASET = 'estaticos_market.csv'
TEST_DATASET1 = 'estaticos_portfolio1.csv'
TEST_DATASET2 = 'estaticos_portfolio2.csv'
TEST_DATASET3 = 'estaticos_portfolio3.csv'

# Number of clusters to use in KMeans
N_CLUSTERS = 22

# Threashold to filter columns
NAN_THRESH = 0.6

# K values for average precision calculus in metadate generation
APK_VALUES = [
    1000,
    5000,
    10000
]

# test_size for metadate generation
TEST_SIZE = 0.2

base_path = path.dirname(path.dirname(squad_3_ad_data_science.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'models')

train_dataset = path.join(data_path, TRAIN_DATASET)
test_datasets = [
    path.join(data_path, TEST_DATASET1),
    path.join(data_path, TEST_DATASET2),
    path.join(data_path, TEST_DATASET3),
]

performance_metadata_path = path.join(workspace_path, 'performance.json')

# Feature engineering configurations

# Data with features selection, NaN fixing and scaled
TRAIN_DATA_PKL = 'pipelined_data.pkl'

# TRAIN_DATA_PKL with clusters
TRAIN_DATA_PKL_LABELED = 'labeled_data.pkl'

# MiniBatchKMeans models pickle
MBK_PICKLE = 'mbk.pkl'

# Columns with True/False values but read as object
TO_FIX_OBJ2BOOL = [
    'fl_passivel_iss',
    'fl_antt',
    'fl_spa',
    'fl_simples_irregular',
    'fl_veiculo',
]

# Columns to just remove
TO_REMOVE = [
    'nm_micro_regiao',              # We use just the macro representation
    'dt_situacao',                  # We use just the macro representation
    'de_natureza_juridica',         # We use just the macro representation
    'de_ramo',                      # We use just the macro representation
    'nm_meso_regiao',               # We use just the macro representation
    'nm_divisao',                   # We use just the macro representation
]

# Variance % that we want to describe with our features.
EXPLAINED_VARIANCE_RATIO = 0.6

# Columns to fix NaN
# '<column_name>': {
#   'method': <mean, median, mode or const>,
#   'value': <string_value>,
# }
NAN_FIXES = {
    'empsetorcensitariofaixarendapopulacao': {
        'method': 'mean',
        'value': None,
    },
    'sg_uf_matriz': {
        'method': 'const',
        'value': 'outro',
    },
    'qt_socios_st_regular': {
        'method': 'mean',
        'value': None,
    },
    'vl_total_veiculos_leves_grupo': {
        'method': 'const',
        'value': 0,
    },
    'vl_faturamento_estimado_aux': {
        'method': 'median',
        'value': None,
    },
    'idade_minima_socios': {
        'method': 'mean',
        'value': None,
    },
    'fl_optante_simples': {
        'method': 'const',
        'value': True,
    },
    'nm_segmento': {
        'method': 'const',
        'value': 'OUTRAS ATIVIDADES DE SERVICOS',
    },
    'qt_socios_pj': {
        'method': 'const',
        'value': 0,
    },
    'vl_faturamento_estimado_grupo_aux': {
        'method': 'mean',
        'value': None,
    },
    'de_saude_tributaria': {
        'method': 'const',
        'value': 'VERDE',
    },
    'de_saude_rescencia': {
        'method': 'const',
        'value': 'ACIMA DE 1 ANO',
    },
    'idade_maxima_socios': {
        'method': 'mean',
        'value': None,
    },
    'vl_total_veiculos_pesados_grupo': {
        'method': 'const',
        'value': 0,
    },
    'qt_socios_pf': {
        'method': 'median',
        'value': None,
    },
    'qt_socios_masculino': {
        'method': 'median',
        'value': None,
    },
    'qt_socios': {
        'method': 'median',
        'value': None,
    },
    'de_faixa_faturamento_estimado': {
        'method': 'const',
        'value': 'DE R$ 81.000,01 A R$ 360.000,00',
    },
    'idade_media_socios': {
        'method': 'median',
        'value': None,
    },
    'setor': {
        'method': 'const',
        'value': 'COMERCIO',
    },
    'nu_meses_rescencia': {
        'method': 'mean',
        'value': None,
    },
    'de_faixa_faturamento_estimado_grupo': {
        'method': 'const',
        'value': 'DE R$ 81.000,01 A R$ 360.000,00',
    },
    'de_nivel_atividade': {
        'method': 'const',
        'value': 'MEDIA',
    },
    'fl_optante_simei': {
        'method': 'const',
        'value': False,
    },
}

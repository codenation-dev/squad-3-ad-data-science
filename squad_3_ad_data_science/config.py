from os import path

import squad_3_ad_data_science

# Datasets to use as test
TRAIN_DATASET = 'estaticos_market.csv'
TEST_DATASET1 = 'estaticos_portfolio1.csv'
TEST_DATASET2 = 'estaticos_portfolio2.csv'
TEST_DATASET3 = 'estaticos_portfolio3.csv'

# Number of clusters to use in KMeans
N_CLUSTERS = 22

# Cluster label for column
CLUSTER_LABEL = 'cluster_label'

# Threashold to filter columns
NAN_THRESH = 0.6

# K values for average precision calculus in metadate generation
APK_VALUES = [
    1000,
    5000,
    10000
]

# test_size for metadata generation
TEST_SIZE = 0.3

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
    'nm_micro_regiao',
    'dt_situacao',
    'de_natureza_juridica',
    'vl_faturamento_estimado_aux',
    'nm_meso_regiao',
    'nm_segmento',
]

# Variance % that we want to describe with our features.
EXPLAINED_VARIANCE_RATIO = 0.6

# Special labels for filtering results
SPECIAL_LABELS = [
    'setor',
    'de_faixa_faturamento_estimado_grupo',
]

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
    'nm_divisao': {
        'method': 'const',
        'value': 'COMERCIO VAREJISTA',
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
    'idade_minima_socios': {
        'method': 'mean',
        'value': None,
    },
    'fl_optante_simples': {
        'method': 'const',
        'value': True,
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

# Manually ordinal encoding, keeping semantics for
# each column.
# {
#   <column_name> {
#       '<old_value>' : <new_value>
#                   .
#                   .
#                   .
#       '<old_value>' : <new_value>
#   }
# }
#
ORDINAL_ENCODE = {
    'de_saude_tributaria': {
        'VERDE': 0,
        'AZUL': 1,
        'AMARELO': 2,
        'CINZA': 3,
        'LARANJA': 4,
        'VERMELHO': 5,
    },
    'de_nivel_atividade': {
        'MUITO BAIXA': 0,
        'BAIXA': 1,
        'MEDIA': 2,
        'ALTA': 3,
    },
    'de_saude_rescencia': {
        'SEM INFORMACAO': 0,
        'ATE 3 MESES': 1,
        'ATE 6 MESES': 2,
        'ATE 1 ANO': 3,
        'ACIMA DE 1 ANO': 4,
    },
    'idade_emp_cat': {
        '<= 1': 0,
        '1 a 5': 1,
        '5 a 10': 2,
        '10 a 15': 3,
        '15 a 20': 4,
        '> 20': 5,
    },
    'de_faixa_faturamento_estimado': {
        'ATE R$ 81.000,00': 0,
        'DE R$ 81.000,01 A R$ 360.000,00': 0,
        'DE R$ 360.000,01 A R$ 1.500.000,00': 1,
        'DE R$ 1.500.000,01 A R$ 4.800.000,00': 1,
        'DE R$ 4.800.000,01 A R$ 10.000.000,00': 1,
        'DE R$ 10.000.000,01 A R$ 30.000.000,00': 2,
        'DE R$ 30.000.000,01 A R$ 100.000.000,00': 2,
        'DE R$ 100.000.000,01 A R$ 300.000.000,00': 2,
        'DE R$ 300.000.000,01 A R$ 500.000.000,00': 3,
        'DE R$ 500.000.000,01 A 1 BILHAO DE REAIS': 3,
        'ACIMA DE 1 BILHAO DE REAIS': 3,
    },
}

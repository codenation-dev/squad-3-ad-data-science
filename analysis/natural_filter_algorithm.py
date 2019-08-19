import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

start_time = time.time()

df_market = pd.read_csv('estaticos_market.csv')
df_portfolio1 = pd.read_csv('estaticos_portfolio1.csv', usecols=['id'])
df_portfolio2 = pd.read_csv('estaticos_portfolio2.csv')
df_portfolio3 = pd.read_csv('estaticos_portfolio3.csv')


# Apply threshold for dropping NaNs with more than 60%

removed_columns = []
null_count = df_market.isna().sum()
percent = null_count/df_market.shape[0]
for col in df_market.iloc[:, 1:]:
    if df_market[col].isna().sum()/df_market[col].shape[0] > 0.6:
        removed_columns.append(col)

new_columns = list(set(df_market.columns) - set(removed_columns))
df_market = df_market.loc[:, new_columns]

# Splitting between ID and variables
df_id = df_market['id']
df_dummy = df_market.drop('id',axis=1)

# Fix True/False objects
df_dummy['fl_passivel_iss'] = df_dummy['fl_passivel_iss'].astype('bool')
df_dummy['fl_antt'] = df_dummy['fl_antt'].astype('bool')
df_dummy['fl_spa'] = df_dummy['fl_spa'].astype('bool')
df_dummy['fl_simples_irregular'] = df_dummy['fl_simples_irregular'].astype('bool')
df_dummy['fl_veiculo'] = df_dummy['fl_veiculo'].astype('bool')

# Filling NaNs in the rest of the Dataset

df_dummy['empsetorcensitariofaixarendapopulacao'] = df_dummy['empsetorcensitariofaixarendapopulacao'].fillna(
    df_dummy['empsetorcensitariofaixarendapopulacao'].mean()) # oreenchendo com a media
df_dummy['sg_uf_matriz'] = df_dummy['sg_uf_matriz'].fillna('outro')
df_dummy['qt_socios_st_regular'] = df_dummy['qt_socios_st_regular'].fillna(
    df_dummy['qt_socios_st_regular'].mean())
df_dummy['empsetorcensitariofaixarendapopulacao'] = df_dummy['empsetorcensitariofaixarendapopulacao'].fillna(
    df_dummy['empsetorcensitariofaixarendapopulacao'].median())
df_dummy['nm_divisao'] = df_dummy['nm_divisao'].fillna('COMERCIO VAREJISTA')
df_dummy['vl_total_veiculos_leves_grupo'] = df_dummy['vl_total_veiculos_leves_grupo'].fillna(0)
df_dummy['vl_faturamento_estimado_aux'] = df_dummy['vl_faturamento_estimado_aux'].fillna(
    df_dummy['vl_faturamento_estimado_aux'].median())
df_dummy['idade_minima_socios'] = df_dummy['idade_minima_socios'].fillna(
    df_dummy['idade_minima_socios'].mean())
df_dummy['fl_optante_simples'] = df_dummy['fl_optante_simples'].fillna(True) # has 20k more readings
df_dummy['nm_segmento'] = df_dummy['nm_segmento'].fillna('OUTRAS ATIVIDADES DE SERVICOS')
df_dummy['qt_socios_pj'] = df_dummy['qt_socios_pj'].fillna(0)
df_dummy['nm_meso_regiao'] = df_dummy['nm_meso_regiao'].fillna('OUTRO')
df_dummy['fl_optante_simei'] = df_dummy['fl_optante_simei'].fillna(True) # majority
df_dummy['vl_faturamento_estimado_grupo_aux'] = df_dummy['vl_faturamento_estimado_grupo_aux'].fillna(
    df_dummy['vl_faturamento_estimado_grupo_aux'].mean()) # distribuição com muita assimetria
df_dummy['de_saude_tributaria'] = df_dummy['de_saude_tributaria'].fillna('VERDE') # assume-se que, ao nao preencher, esta no verde
df_dummy['de_saude_rescencia'] = df_dummy['de_saude_rescencia'].fillna('ACIMA DE 1 ANO')
df_dummy['idade_maxima_socios'] = df_dummy['idade_maxima_socios'].fillna(df_dummy['idade_maxima_socios'].mean())
df_dummy['vl_total_veiculos_pesados_grupo'] = df_dummy['vl_total_veiculos_pesados_grupo'].fillna(0)
df_dummy['qt_socios_pf'] = df_dummy['qt_socios_pf'].fillna(df_dummy['qt_socios_pf'].median())
df_dummy['qt_socios_masculino'] = df_dummy['qt_socios_masculino'].fillna(
    df_dummy['qt_socios_masculino'].median()) # talvez preencher com 0 seja melhor
df_dummy['qt_socios'] = df_dummy['qt_socios'].fillna(
    df_dummy['qt_socios'].median())
df_dummy['de_faixa_faturamento_estimado'] = df_dummy['de_faixa_faturamento_estimado'].fillna(
    'DE R$ 81.000,01 A R$ 360.000,00')
df_dummy['nm_micro_regiao'] = df_dummy['nm_micro_regiao'].fillna('NAO ESPECIFICADO')
df_dummy['idade_media_socios'] = df_dummy['idade_media_socios'].fillna(df_dummy['idade_media_socios'].median())
df_dummy['setor'] = df_dummy['setor'].fillna('COMERCIO')
df_dummy['nu_meses_rescencia'] = df_dummy['nu_meses_rescencia'].fillna(df_dummy['nu_meses_rescencia'].mean())
#df_dummy['de_faixa_faturamento_estimado_grupo'] = df_dummy['de_faixa_faturamento_estimado_grupo'].fillna(
#    'DE R$ 81.000,01 A R$ 360.000,00') - dropped afterwards
df_dummy['de_nivel_atividade'] = df_dummy['de_nivel_atividade'].fillna('MEDIA')
# df_dummy['dt_situacao'] = df_dummy['dt_situacao'].fillna('2005-11-03') - dropped afterwards


# Ordinal encoding - manually

df_ord = df_dummy

df_ord['de_saude_tributaria'] = df_ord['de_saude_tributaria'].replace('VERDE',0)
df_ord['de_saude_tributaria'] = df_ord['de_saude_tributaria'].replace('AZUL',1)
df_ord['de_saude_tributaria'] = df_ord['de_saude_tributaria'].replace('AMARELO',2)
df_ord['de_saude_tributaria'] = df_ord['de_saude_tributaria'].replace('CINZA',3)
df_ord['de_saude_tributaria'] = df_ord['de_saude_tributaria'].replace('LARANJA',4)
df_ord['de_saude_tributaria'] = df_ord['de_saude_tributaria'].replace('VERMELHO',5)

df_ord['de_nivel_atividade'] = df_ord['de_nivel_atividade'].replace('MUITO BAIXA',0)
df_ord['de_nivel_atividade'] = df_ord['de_nivel_atividade'].replace('BAIXA',1)
df_ord['de_nivel_atividade'] = df_ord['de_nivel_atividade'].replace('MEDIA',2)
df_ord['de_nivel_atividade'] = df_ord['de_nivel_atividade'].replace('ALTA',3)

df_ord['de_saude_rescencia'] = df_ord['de_saude_rescencia'].replace('SEM INFORMACAO',0)
df_ord['de_saude_rescencia'] = df_ord['de_saude_rescencia'].replace('ATE 3 MESES',1)
df_ord['de_saude_rescencia'] = df_ord['de_saude_rescencia'].replace('ATE 6 MESES',2)
df_ord['de_saude_rescencia'] = df_ord['de_saude_rescencia'].replace('ATE 1 ANO',3)
df_ord['de_saude_rescencia'] = df_ord['de_saude_rescencia'].replace('ACIMA DE 1 ANO',4)

df_ord['idade_emp_cat'] = df_ord['idade_emp_cat'].replace('<= 1',0)
df_ord['idade_emp_cat'] = df_ord['idade_emp_cat'].replace('1 a 5',1)
df_ord['idade_emp_cat'] = df_ord['idade_emp_cat'].replace('5 a 10',2)
df_ord['idade_emp_cat'] = df_ord['idade_emp_cat'].replace('10 a 15',3)
df_ord['idade_emp_cat'] = df_ord['idade_emp_cat'].replace('15 a 20',4)
df_ord['idade_emp_cat'] = df_ord['idade_emp_cat'].replace('> 20',5)

# For the estimated income, better results with larger chunks rather than ordinal encoding
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('ATE R$ 81.000,00',0)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 81.000,01 A R$ 360.000,00',0)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 360.000,01 A R$ 1.500.000,00',1)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 1.500.000,01 A R$ 4.800.000,00',1)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 4.800.000,01 A R$ 10.000.000,00',1)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 10.000.000,01 A R$ 30.000.000,00',2)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 30.000.000,01 A R$ 100.000.000,00',2)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 100.000.000,01 A R$ 300.000.000,00',2)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 300.000.000,01 A R$ 500.000.000,00',3)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('DE R$ 500.000.000,01 A 1 BILHAO DE REAIS',3)
df_ord['de_faixa_faturamento_estimado'] = df_ord['de_faixa_faturamento_estimado'].replace('ACIMA DE 1 BILHAO DE REAIS',3)


# Get dummies
df_new = pd.get_dummies(df_ord.drop(['nm_micro_regiao','dt_situacao','de_natureza_juridica','vl_faturamento_estimado_aux',
                                       'nm_meso_regiao','de_faixa_faturamento_estimado_grupo','nm_segmento','setor'],
                                      axis=1),sparse=True,drop_first=True)

# Standardize all values in the dataset (mean=0, std=1)
df_scaled = StandardScaler().fit_transform(df_new)
df_scaled = pd.DataFrame(df_scaled,columns=df_new.columns)

# Principal Component Analysis

pca = PCA()
pca.fit_transform(df_scaled)
ratio = pca.explained_variance_ratio_

counter = 0
value = 0

while value <= 0.6: # decided to use 60% of the variance according to Multivariate Analysis book
  value = np.cumsum(ratio)[counter]
  counter += 1
num_components = int(counter)

df_reduced = pd.DataFrame(PCA(n_components=num_components).fit_transform(df_scaled))

def sector_filtering(portfolio):

    df_reduced['labels'] = OrdinalEncoder(cols=['setor']).fit_transform(df_dummy['setor'])
    df_reduced['labels2'] = OrdinalEncoder(cols=['de_faixa_faturamento_estimado_grupo']).fit_transform(df_dummy['de_faixa_faturamento_estimado_grupo'])

    X = pd.concat([df_id,df_reduced], axis='columns')

    # portfolio information
    pf_filled = X.loc[X['id'].isin(portfolio['id'].values)]

    # part of the market that shares the same clusters

    pf_out = X.loc[X['labels'].isin(list(pf_filled['labels'].unique()))]
    pf_out = pf_out.loc[X['labels2'].isin(list(pf_filled['labels2'].unique()))]

    # customer that are not yet on the company's portfolio
    sample = pf_filled.iloc[:,:num_components-1].sample(frac=0.7, random_state=42) # num_comp-1 for it not to account for the labels in the dot product

    pf_rec = pf_out.loc[~pf_out['id'].isin(sample['id'])]
    pf_rec = pf_rec.iloc[:,:num_components-1] # num_comp-1 for it not to account for the labels in the dot product

    cosine_sim = cosine_similarity(pf_rec.drop(['id'],axis='columns'),sample.drop(['id'],axis='columns'))
    cosine_sim = np.sum(cosine_sim, axis=1) # best results with sum. amax and mean already tested

    pf_rec['score'] = list(cosine_sim)

    # list new leads to recommend
    market = list(pf_rec.sort_values('score', ascending=False)['id'])
    test = list(pf_filled.loc[~pf_filled['id'].isin(sample['id'])]['id'])

    return market, test
'''
KMeans and Hybrid methods tested but WORSE results
def mini_kmeans(number_of_clusters, data):
    model = MiniBatchKMeans(n_clusters = number_of_clusters, batch_size=500)
    labels = model.fit_predict(data)
    return labels
def kmeans_method(portfolio):
    df_reduced['labels'] = mini_kmeans(20,df_reduced)
    # df_reduced['labels2'] = OrdinalEncoder(cols=['setor']).fit_transform(df_dummy['setor'])
    # df_reduced['labels10'] = OrdinalEncoder(cols=['de_faixa_faturamento_estimado_grupo']).fit_transform(df_dummy['de_faixa_faturamento_estimado_grupo'])

    X = pd.concat([df_id,df_reduced], axis='columns')

    # portfolio 1 information
    pf1_filled = X[X['id'].isin(portfolio['id'].values)]

    # part of the market that shares the same clusters
    pf1_out = X.loc[X['labels'].isin(list(pf1_filled['labels'].unique()))]
    # pf1_out = pf1_out.loc[X['labels2'].isin(list(pf1_filled['labels2'].unique()))]
    # pf1_out = pf1_out.loc[X['labels10'].isin(list(pf1_filled['labels10'].unique()))]

    # customer that are not yet on the company's portfolio
    sample = pf1_filled.sample(frac=0.7, random_state=42)

    counting = pd.concat([pd.DataFrame(pf1_filled['labels'].value_counts().index.tolist()),
                    pd.DataFrame(pf1_filled['labels'].value_counts().tolist())], axis=1)
    counting.columns = ['labels','values']
    counting = counting.sort_values('labels')

    pf1_rec = pf1_out.loc[~pf1_out['id'].isin(sample['id'])]

    cosine_sim = cosine_similarity(pf1_rec.drop(['id'],axis='columns'),sample.drop(['id'],axis='columns'))
    cosine_sim = np.sum(cosine_sim, axis=1)

    pf1_rec = pf1_rec.merge(counting, on='labels', how='left')

    pf1_rec['score'] = list(cosine_sim)

    # multiplication factor

    pf1_rec['multiplier'] = pf1_rec['score'] * np.sqrt(pf1_rec['values'])

    # list of new leads to recommend

    market = list(pf1_rec.sort_values('score', ascending=False)['id'])
    test = list(pf1_filled.loc[~pf1_filled['id'].isin(sample['id'])]['id'])

    return market, test
'''

def custom_apk(market_list, test_list, k):

    '''Custom function for the average precision at k - AP@k

    market_list: list of ids from all the recommended companies sorted by descending score (order does matter)

    test_list: list of ids from all companies in the ~20% test dataframe (order does not matter)

    k: number of recommendations'''

    if len(market_list) > k:
        market_list = market_list[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(market_list):
        if p in test_list:
          num_hits += 1.0
          score += num_hits / (i + 1.0)

    return score / len(test_list)

market1, test1 = sector_filtering(df_portfolio1)
onek, fivek, tenk = custom_apk(market1, test1, 1000),custom_apk(market1, test1, 5000),custom_apk(market1, test1, 10000)

market2, test2 = sector_filtering(df_portfolio2)
onek2, fivek2, tenk2 = custom_apk(market2, test2, 1000),custom_apk(market2, test2, 5000),custom_apk(market2, test2, 10000)

market3, test3 = sector_filtering(df_portfolio3)
onek3, fivek3, tenk3 = custom_apk(market3, test3, 1000),custom_apk(market3, test3, 5000),custom_apk(market3, test3, 10000)

elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time}')
print(' ')
print('Portfolio #1')
print(' ')
print(f'AP@1k: {onek}')
print(f'AP@5k: {fivek}')
print(f'AP@10k: {tenk}')
print(' ')
print('Portfolio #2')
print(' ')
print(f'AP@1k: {onek2}')
print(f'AP@5k: {fivek2}')
print(f'AP@10k: {tenk2}')
print(' ')
print('Portfolio #3')
print(' ')
print(f'AP@1k: {onek3}')
print(f'AP@5k: {fivek3}')
print(f'AP@10k: {tenk3}')
print(' ')
print(f'Number of components used : {num_components}')

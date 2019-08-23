# Model report - [`squad-3-ad-data-science`]
This report should contain all relevant information regarding your model. Someone reading this document should be able to easily understand and reproducible your findings.

## Checklist
Mark which tasks have been performed

- [ ] **Summary**: you have included a description, usage, output,  accuracy and metadata of your model.
- [ ] **Pre-processing**: you have applied pre-processing to your data and this function is reproducible to new datasets.
- [ ] **Feature selection**: you have performed feature selection while modeling.
- [ ] **Modeling dataset creation**: you have well-defined and reproducible code to generate a modeling dataset that reproduces the behavior of the target dataset. This pipeline is also applicable to generate the deploy dataset.
- [ ] **Model selection**: you have chosen a suitable model according to the project specification.
- [ ] **Model validation**: you have validated your model according to the project specification.
- [ ] **Model optimization**: you have defined functions to optimize hyper-parameters and they are reproducible.
- [ ] **Peer-review**: your code and results have been verified by your colleagues and pre-approved by them.
- [ ] **Acceptance**: this model report has been accepted by the Data Science Manager. State name and date.

## Summary

The model is a `lead recommendation system`, it is based on cosine similarity to generate a pairwise similarity score metric between multiple companies.   

### Usage

1. clone the repository
2. download test dataset from cloud storage
3. run `mkdir workspace/data`
4. run `mkdir workspace/models`
5. configure train data on `config.py`
6. install [Docker](https://www.docker.com/)
7. run `make run`
8. run `make predict INPUT='<input_filepath>'`

### Output

##### Domain

is all `id`s present on `estaticos_market.csv`

##### Output

will be subset of `id`s of `estaticos_market.csv`


#### Metadata


[Metadata](../squad_3_ad_data_science/project_metadata.json)

#### Coverage

-

### Performance Metrics

| metric            | `'portfolio1'`| `'portfolio2'` | `'portfolio3'` |
| ------------------| ------------- | -------------- | -------------- |
| Average precision | .0            | .12            | .12            |
| n of companies    | .98           | .8             | .9             |

## Pre-processing

1. remove columns listed on `config.TO_REMOVE`: most of then represents redundant information on dataset, as described on [dict.json](../analysis/dict.json)
2. remove columns with `NaN` rate > `config.NAN_THRESH`, configured initialy as 0.6 (60%): columns coudl no be well treated by any filling
3. Fix columns from `config.TO_FIX_OBJ2BOOL`: they was `True/False` values but read as string
4. Impute remaining `NaN`s as set on `config.NAN_FIXES`: manual analysis/testing
5. Encode columns on `config.ORDINAL_ENCODE` manually: encode ordering by semantic (like the worst value 0 and best N)
6. Encode columns from `config.SPECIAL_LABELS` and store to join after feature selection: some columns witch by tests we have the hipotesis that can be use to filter results
7. Scale data: to use clustering strategies
8. Use PCA to select components to describe 60% of variance

## Feature selection

We removed some columns (`config.TO_REMOVE`) by information redundance and used PCA to reduce dataset dimension by a n_components that represents 60% of variance

## Modeling

We tried to use `KMeans` to improve recomendations, but after some tests we decided to let it just as a option on software predicting. 

Our recomendations comes mainly from `cosine_similarity` calculated between user's portfolio and the rest of the market.

### Model selection

`KMeans` clustering to select possible recomendations was our first guess, but after some tests, a filter based on some analysis over features and tests with our validation metric, we choose to use as default just filtering methods and `cosine_similarity` to order the leads.

### Model validation

1. Train dataset `estaticos_market.csv` was pipelined
2. Test files `estaticos_portfolio<1, 2, 3>.csv` was divided into train and test (70/30)
3. We use average precision (implemented on `squad_3_ad_data_science.validation`) to measure our recomendations quality and make manual filter to maximize this metrix
 
### Model optimization

By tests we tried to find features that could represent the preferences from the portfolios, using then to filter results or clusterize dataset

## Additional resources
-
# Project specification - [`squad-3-ad-data-science`]
> This document contains a data-science oriented description of the project. It orients the data science project towards data and technology. It serves the purposes of

* outlining a solution,
* defining the scope of the project,
* defining measurable success metrics,
* defining interfaces to the production environment,
* gathering information regarding relevant datasets and features to the problem,
* upstream communication and agreement on requisites.


## Checklist
> Mark which tasks have been performed.

- [ ] **Summary**: all major specs are defined including success criteria, scope, etc.
- [ ] **Output specification**: the output format, metadata and indexation have been defined and agreed upon.
- [ ] **Solution architecture**: a viable and minimal architecture for the final solution has been provided and agreed upon.
- [ ] **Limitations and risks**: a list with the main limitations and risks associated to the project has been provided.
- [ ] **Related resources**: lists with related datasets, features and past projects have been given.
- [ ] **Peer-review**: you discussed and brainstormed with colleagues the outlined specifications.
- [ ] **Acceptance**: these specifications have been accepted by the Data and Product directors. State names and date.

## Summary
> The table below summarizes the key requirements for the project.

| problem type                             | target population | entity | N_target | min_coverage | N_labeled | sucess_metrics      | updt_freq   |
|------------------------------------------|-------------------|--------|----------|--------------|-----------|---------------------|-------------|
| content based filter for recommendations | business          | CNPJ   |  -       | 80%          | NA        | average precision   | as new companies come in dataset     |


### Objective

This project aims at developing a model to recommend leads for a company based on its previous clients.

### Target population

| Entity | Region | Type        | Size | Status | Sector   | N_target |
|--------|--------|-------------|------|--------|----------|----------|
| CNPJ   | Brasil | companies   | any  | active | -        | -        |


#### Subsetting

Variables present in `config.SPECIAL_LABELS` are used to filter market to make recomendations. Model removes from recomendations possibilities all rows that do not share its special labels with at least one row of user's portfolio for each label.

Initially used special labels:
| Label                                 | Based on      |
|---------------------------------------|---------------|
| `setor`                               | Tests & business knowledge (little)        | 
| `de_faixa_faturamento_estimado_grupo` | Tests         |

`predict` function offer option to do not use special labels on recomendation.

### Output specification

The model will outputs a list of string that will be a subset of `id` column of input file.

#### Metadata

[Metadata file](../squad_3_ad_data_science/project_metadata.json)

### Problem type

As the problem is make recomendations for a user with a portfolio, based only on it's portfolio. So, the model should be a content-based filter, validated by metrics that analyze results of recomendations.

## Solution architecture

The model should consume data from total market's dataset, restricting input `id`s possiblities by market dataset's `id`s. 

For current version, user must put `estaticos_market.csv` train file and and `estaticos_portfolio<1, 2, 3>.csv`  test files on data folder.

To generate predictions, put input file on predict folder and run `make predict INPUT='<path_to_file>'`. 

### Validation metric (Average Precision)

For our recommendation system, we have divided the portfolios datasets into train and test, considered precision and recall as follows:

<a href="#image-1"><img src="https://latex.codecogs.com/png.latex?Precision&space;=&space;\frac{number\;&space;of\;&space;test\;&space;leads\;&space;inside\;&space;the\;&space;recommendations}{total\;&space;number\;&space;of\;&space;recommendations}" title="Precision Formula" id="image-3"/></a>

<a href="#image-2"><img src="https://latex.codecogs.com/png.latex?Recall&space;=&space;\frac{number\;&space;of\;&space;test\;&space;leads\;&space;inside\;&space;the\;&space;recommendations}{total\;&space;number\;&space;of\;&space;test\;&space;subset}" title="Recall Formula" id="image-3"/></a>

Then, we have used the Average Precision at k (AP@k) on each portfolio, to valuate and compare our models, where k is the number of recommendations requested, from rank 1 through k.

<a href="#image-3"><img src="https://latex.codecogs.com/png.latex?AP@k&space;=&space;\sum_{i=1}^{k}(precision\;&space;at&space;\;\mathbf{i})\cdot&space;(change\;&space;in\;&space;recall\;&space;at\;&space;\mathbf{i})&space;=&space;\sum_{i=1}^{k}&space;P(i)\Delta&space;r(i)" title="Average Precision at k Formula" id="image-3"/></a>

We have used the Average Precision because it gives us an idea of not only the number of leads from the test subset inside the recommendations, but also the position of those leads.

### Limitations and risks

| Limitation                              | Likelihood | Loss                               | Contingency                        |
|-----------------------------------------|------------|------------------------------------|------------------------------------|
| User's portfolio contains new ids           | 100%       | not possible make prediction | Update market list |
| User's portfolio with high variance | 50%        | predictions could be near to random          | -            |



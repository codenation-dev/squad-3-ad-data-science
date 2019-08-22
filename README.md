# [`squad-3-ad-data-science`]

This model is a `recomendation system` that analyze a market to recommend leads to users with a previous list of clients that are inside the market.

## Stakeholders

| Role           | Responsibility | Full name                      | e-mail                           |
| -------------- | -------------- | ------------------------------ | -------------------------------- |
| Data Scientist | Author         | [`Lincoln Vinicius Schreiber`] | [`lincolnschreiber@gmail.com`]   |
| Data Scientist | Author         | [`Murilo Menezes Mendonça`]    | [`murilommen@gmail.com`]         |
| Data Scientist | Author         | [`Nathan dos Santos Nunes`]    | [`nathan.sn@hotmail.com`]        |
| Data Scientist | Author         | [`Thiago Sant' Helena`]        | [`thiago.sant.helena@gmail.com`] |

## Usage

Usage is standardized across models. There are two main things you need to know, the development workflow and the Makefile commands.

Both are made super simple to work with Git and Docker while versioning experiments and workspace.

All you'll need to have setup is Docker and Git, which you probably already have. If you don't, feel free to ask for help.

Makefile commands can be accessed using `make help`.

Make sure that **docker** is installed.

Clone the project from the analytics Models repo.
```
git clone https://github.com/<@github_username>/squad-3-ad-data-science.git
cd squad-3-ad-data-science
mkdir workspace/data
mkdir workspace/models
```

Be sure to configure train and test data on `squad_3_ad_data_science/config.py`, placing files on `workspace/data` folder.

To train and generate test metadata, run 
```
make run
```

Performance metadata available on `workspace/performance.json`

To get recomendations, run
```
make predict INPUT='<path_to_input_file_with_column_of_ids>' PARAMS='--k 10'
```

Param `k` stands for the number of recomendations. Default is 10.


## Final Report (to be filled once the project is done)

### Model Frequency

`make run` takes ~5 min (assuming installed libraries)

`make predict` takes less than 1min


### Model updating

More filtering and feature tunning can be added to `feature_engineering.py`, more validation functions to generate extra metadata for model validation on `validation.py` and more recomendation functions on `recomendations.py`.

`main.py` must be updated to apply new methods.

Keep logs with `loguru`.

### Maintenance

To deploy as web application, use functions implemented on module to make predicts with `main.predict(input_file, **kwargs)`. 

### Minimum viable product

--

### Early adopters

--

## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Explain you folder strucure

* [docs](./docs): contains documentation of the project
* [analysis](./analysis/): contains notebooks of data and modeling experimentation.
* [tests](./tests/): contains files used for unit tests.
* [squad_3_ad_data_science](./squad_3_ad_data_science/): main Python package with source of the model.

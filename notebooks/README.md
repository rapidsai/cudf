# Notebook

## Create conda environment for the notebook

The following instructions are tested on 64-bit linux only.

The following create an environment named "pycudf_notebook_py35" with all the necessary packages:

```bash
$ conda env create -f=../conda_environments/notebook_py35.yml
```

Or with a different environment name:


```bash
$ conda env create -f=../conda_environments/notebook_py35.yml -n <new_env_name>
```

Continue with the following steps using the conda environment
created.

```bash
source activate pycudf_notebook_py35
```

## Launch notebook


```bash
$ jupyter notebook
```


## Import data into MapD

Decompress the `ipums_easy.csv.gz` for sample data used by the notebook.

Using `mapdql`:

1. create table `ipums_easy` with the content in
   `./create_table_ipums_easy.txt`

2. import CSV

    ```SQL
    COPY ipums_easy FROM './ipums_easy.csv';
    ```


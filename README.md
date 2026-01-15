## Overall Structure
It's not set in stone yet.

- analysis/         
  - run.py         (Main script to run the tool)
- pre_analysis/    (Pre-analysis scripts for generating static type analysis results)
- evaluation/      (Evaluation scripts for generating and printing results)
  - eval.py        (Evaluate the results)
  - print_table.py (Print the results in a table format)
- run/             (The code for main components)
  - make_data.py   (Generate the data for the type similarity model)
  - rerank.py      (Implementation for Re-ranking and Augmentation)
  - run_static_analyis.py (Run static analysis tools)
- train.py        (Train the type similarity model for BetterTypes4Py dataset)
- train_many4types.py (Train the type similarity model for ManyTypes4Py dataset)


## Installation and Setup

You can use the provided Docker image to evaluate the results.
```bash
docker build -t typecare .
```

Then, you can run the Docker container with the following command:
```bash
docker run -it typecare
```

In the container, you have to install a requirement library:
- [libcst](https://libcst.readthedocs.io/en/latest/#) (v0.4.2)

You can install libcst by the following command:

```bash
# in the Docker container
. "$HOME/.cargo/env"
rustc --version # rustc 1.92.0 ...

pip install libcst==0.4.2
```

We provided pre-computed outputs to save the time.
To reproduce the results, you can run the following command:
```bash
# in the Docker container
cd TypeCare

# Download pre-computed outputs
wget ...
7z x data.7z
```

The pre-computed outputs were organized as follows:
- `predictions` It contains outputs of existing language models (TypeT5, TypeGen, TIGER) and our type similarity models (Related to [this section](#train-the-type-similarity-model)).
- `data` It contains outputs of static analysis tools when annotating types created by the models (Related to [this section](#run-pre-analysis)). 
- `output` It contains outputs for an evaluation (Related to [this section](#run-re-ranking--augmentation)). 

## Evaluation in the Docker Container

Using the pre-computed outputs, you can evaluate results through the following command:
```bash
# Evaluate pre-computed outputs
python -m analysis.run --evaluate
```
Please note that the initial run may take a while as it will download tokenizers.
Then, you can obtain main reulsts of Table 3~7 in the paper such as follows:

```bash
=== Main Table ===
  Model     Exact T1(%)    Exact T3(%)    Exact T5(%)     Base T1(%)     Base T3(%)     Base T5(%)
-------  --------------  -------------  -------------  -------------  -------------  -------------
 TypeT5           71.4%          77.2%          78.9%          78.1%          83.7%          85.4%
  +Ours  (+13.6%) 81.1%  (+7.4%) 82.9%  (+5.8%) 83.5%  (+9.2%) 85.3%  (+4.3%) 87.3%  (+3.0%) 88.0%
-------  --------------  -------------  -------------  -------------  -------------  -------------
  Tiger           67.8%          78.0%          80.2%          75.8%          85.5%          88.0%
  +Ours  (+11.8%) 75.8%  (+4.4%) 81.4%  (+4.5%) 83.8%  (+6.5%) 80.7%  (+2.3%) 87.5%  (+2.7%) 90.4%
-------  --------------  -------------  -------------  -------------  -------------  -------------
TypeGen           65.4%          73.4%          75.0%          71.6%          79.9%          81.6%
  +Ours  (+12.5%) 73.6%  (+7.6%) 79.0%  (+6.8%) 80.1%  (+9.9%) 78.7%  (+5.4%) 84.2%  (+4.9%) 85.6%

=== Function Signature Table ===
... (continue with the rest of the results)
```

## Reproduction Results

You can also reproduce the pre-computed results.
From this section, I will introduce steps to reproduce datas.

### Download Benchmarks

To run the analysis, you need to prepare the data.
- [ManyTypes4Py](https://github.com/saltudelft/many-types-4-py-dataset)
- [BetterTypes4Py](https://github.com/utopia-group/TypeT5)

However, some datas were not accessible because repositories were deleted.
Thus, we provided datas used in our experiments via [Hugging Face platform](https://huggingface.co/datasets/marinelay/TypeCare-Datasets).
You can see datasets repository [here](https://huggingface.co/datasets/marinelay/TypeCare-Datasets).

You are able to download the datasets via the following commands:

```bash
# in docker container
cd /home/TypeCare

# Download datasets
hf download --repo-type dataset marinelay/TypeCare-Datasets --local-dir .
tar -I 'zstd -T8' -xvf BetterTypes4Py.tar.zst
tar -I 'zstd -T8' -xvf ManyTypes4Py.tar.zst
```

### Kick the Tire

...

### Reproduction Steps

#### Run pre-analysis

**This section will create `data` contents of the pre-computed result**

TypeCare uses results of static type analysis when annotating types created by the language models. We use the tool, named [pyright](https://github.com/microsoft/pyright), to perform static type analysis.

Thus, we provided scripts to do this step:

```bash
# in docker container
cd /home/TypeCare
python -m pre_analysis.run --tool=<typet5|tiger|typegen>
```

It makes the result of static type analysis in the directory `data`.

#### Train the Type Similarity Model

**This section will create `predictions/typesim_model` contents of the pre-computed result**

You can train the type similarity model using the following command:
```bash
python train.py
python train_many4types.py
```
#### Run Re-ranking & Augmentation

**This section will create `output` contents of the pre-computed result**

Now, you can refine the model answers using the trained type similarity model:
```bash
python -m analysis.run --all
```
If you want to run specific tools, you can specify them using the `--tool` argument:
```bash
python -m analysis.run --tool=<typet|typegen|tiger>
```
Then, the analysis will be performed and the results will be saved in the `output` directory.


### Data
To run the analysis, you need to prepare the data.
- [ManyTypes4Py](https://github.com/saltudelft/many-types-4-py-dataset)
- [BetterTypes4Py](https://github.com/utopia-group/TypeT5)

#### Download ManyTypes4Py
```
cd ~
wget -O ManyTypes4PyDataset-v0.7.tar.gz "https://zenodo.org/records/5244636/files/ManyTypes4PyDataset-v0.7.tar.gz?download=1"
tar -xvzf ManyTypes4PyDataset-v0.7.tar.gz
```


You can download the datasets and place them in the `repos/` directory.

### Run Tool

#### 1-1. Download Pre-Data to run the tool
You can download the pre-computed data files from the following links: https://figshare.com/s/ee0936a79dc01992a1ea

#### 1-2. Generate Pre-Data
If you want to generate the pre-computed data files, you can run the following command.

##### Pre-analysis
To run our tool, you first run pre-analysis to generate results of static type analysis:
```bash
python -m pre_analysis.tiger
python -m pre_analysis.typet5
python -m pre_analysis.typegen
```
This script will generate the necessary data files in the `data/` directory.
Note that the script execution may take a while.

##### Train the Type Similarity Model
You can train the type similarity model using the following command:
```bash
python train.py
python train_many4types.py
```
This will create the two models for each dataset in the `results/typesim_model/` directory.

### 2. Refine the model Answers
Now, you can refine the model answers using the trained type similarity model:
```bash
python -m analysis.run --all
```
If you want to run specific tools, you can specify them using the `--tool` argument:
```bash
python -m analysis.run --tool [typet5,typegen,tiger]
```
Then, the analysis will be performed and the results will be saved in the `output/` directory.

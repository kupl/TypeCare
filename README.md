---
license: mit

doi: 10.57xxx/xxxxx.xxxxx
---

# Artifact for TypeCare (ICSE 2026)

This repository provides the replication package for the ICSE 2026 paper: **"TypeCare: Boosting Python Type Inference Models via Context-Aware Re-Ranking and Augmentation"**([pre-print](https://prl.korea.ac.kr/papers/icse26typecare.pdf))

## Purpose & Artifact Badges

Source Code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18263563.svg)](https://doi.org/10.5281/zenodo.18263563) 
Dataset: [![Hugging Face DOI](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DOI-orange)](https://doi.org/10.57967/hf/7547)

* **Available:** The artifact is persistently archived on Zenodo.
* **Functional:** We provide a comprehensive Docker environment, a minimal working example ([Kick-the-Tire](#kick-the-tire-small-example)), and step-by-step instructions to verify that the tool is operational and correctly evaluates experimental results.
* **Reusable:** Our package includes detailed documentation and modular scripts, allowing researchers to not only reproduce the results in the paper but also easily adapt TypeCare to new datasets (such as example code).

## Overall Structure

#### Source Code

- analysis/         
  - run.py         (Main script to run the tool)
- pre_analysis/    
  - run.py         (Pre-analysis scripts for generating static type analysis results)
- evaluation/      (Evaluation scripts for generating and printing results)
  - eval.py        (Evaluate the results)
  - print_table.py (Print the results in a table format)
- run/             (The code for main components)
  - make_data.py   (Generate the data for the type similarity model)
  - rerank.py      (Implementation for Re-ranking and Augmentation)
  - run_static_analyis.py (Run static analysis tools (pyright))
- train_model/
  - train.py            (Train the type similarity model for BetterTypes4Py dataset)
  - train_many4types.py (Train the type similarity model for ManyTypes4Py dataset)

## System Environment

The experimental results reported in the paper were obtained using the **Main Server** environment. Additionally, we verified that TypeCare's re-ranking and augmentation algorithms are lightweight enough to run on a **Laptop**.

**Main Experimental Setup (Server)**
Component | Specification |
| :---: | :--- |
| CPU | Intel(R) Xeon(R) Silver 4214 |
| RAM | 128GB |
| OS | Ubuntu 22.04 LTS |

**Lightweight Execution Setup (Laptop)**
Component | Specification |
| :---: | :--- |
| CPU | iMac Apple M1 |
| RAM | 16GB |
| OS | macOS Sequoia |

**Note: Compatibility Issue with Apple Silicon (M-series)** 
Currently, there is a known issue where pyright causes errors when building Docker images on Apple Silicon. We recommend **building the image in a Linux environment** as a workaround until this issue is resolved in a future update.

(Despite this issue, you can still view the evaluation tables using the provided pre-computed data.)

### Common Requirements

- Storage: At least **20GB** of available disk space is required to store the datasets and pre-computed outputs.

## Installation and Setup

### Docker Setup

We recommend using the provided Docker image for a consistent evaluation environment.
```bash
git clone https://github.com/kupl/TypeCare.git
cd TypeCare

# Build the Docker image
docker build --no-cache --platform=linux/amd64 -t typecare .

# Run the container
docker run -it typecare
```

### Requirements

Within the container, ensure the following library is installed.
- [libcst](https://libcst.readthedocs.io/en/latest/#) (v0.4.2)
- [pyright](https://github.com/microsoft/pyright)

```bash
# in the Docker container

# Install libcst
. "$HOME/.cargo/env"
rustc --version # rustc 1.92.0 ...
pip install libcst==0.4.2

# Install pyright
pip install pyright
pyright --version

# Expected Output
* Install prebuilt node ..... done.
pyright 1.1.xxx
```

### Pre-computed Outputs

To save time during evaluation, we provided pre-computed outputs.
```bash
# in the Docker container
cd /home/TypeCare

# Download and extract pre-computed outputs
wget https://github.com/kupl/TypeCare/releases/download/v1.0.1/pre_computed.tar.zst
tar -xvf pre_computed.tar.zst
```

**The pre-computed data is organized as follows:**
- `prediction/` Contains outputs from baseline models ([TypeT5](https://github.com/utopia-group/TypeT5), [TypeGen](https://github.com/JohnnyPeng18/TypeGen), [TIGER](https://github.com/cs-wangchong/TypeInfer-Replication)) and our type similarity models.
- `data/` Contains static analysis results generated when annotating types predicted by the models.
- `output/` Final refined outputs used for evaluation.

## Kick the Tire (Small Example)

We provide a minimal example to illustrate the workflow. You can better understand how TypeCare operates by following the code provided below. The target code is located in `example/example.py`.

<details>
<summary>Show/Hide Code</summary>

```python
def foo(x: int):
    return x + 1
```
</details>
</br>

We assume a learning-based model has generated the top-10 candidates, as shown in `prediction/Example/transformed_result.json`.` In this example, the correct type int is initially ranked 3rd.

<details>
<summary>Show/Hide Output</summary>

```json
[
    {
        "repo_name": "example",
        "file_path": "example.py",
        "target": "foo",
        "params": [
            "x"
        ],
        "predictions": [
            [
                "str"
            ],
            [
                "list"
            ],
            [
                "int"
            ],
            [
                "bool"
            ],
            [
                "tuple"
            ],
            [
                "set"
            ],
            [
                "float"
            ],
            [
                "bytes"
            ],
            [
                "None"
            ],
            [
                "dict"
            ]
        ],
        "total_predictions": null,
        "expects": [
            "int"
        ],
        "cat": "builtins",
        "generic": true
    }
]
```
</details>
</br>

1. **Run pre-analysis:** Generate static analysis (pyright) results for the candidates.
```bash
# in the Docker Container
cd /home/TypeCare
python -m pre_analysis.run --tool="example"
```

**Note:** The initial run may take a few minutes to download the required tokenizers.

It annotates candidates as type of parameter `x`, runs static analysis, and finally saves results in `data/Example`.

2. **Run TypeCare:** Apply the Re-ranking & Augmentation algorithm.
```bash
python -m analysis.run --tool="example"
```

**Expected Output:** The log will show that the rank of the correct type (int) has been improved:
```bash
...
Top K
+-------+--------+------------+-------+-----------+------+
| Rank  | Before | Before (%) | After | After (%) | Diff |
+-------+--------+------------+-------+-----------+------+
| Top 1 |   0    |    0.0%    |   1   |  100.0%   | inf% |
| Top 3 |   1    |   100.0%   |   1   |  100.0%   | 0.0% |
| Top 5 |   1    |   100.0%   |   1   |  100.0%   | 0.0% |
+-------+--------+------------+-------+-----------+------+
```
As shown above, TypeCare successfully re-ranks the correct candidate from 3rd to 1st.

## Evaluation 

To evaluate the pre-computed outputs and reproduce the main tables (Tables 3–6) from the paper, run:
```bash
# in the Docker container
cd /home/TypeCare

# Evaluate pre-computed outputs
python -m analysis.run --evaluate
```

```bash
=== Main Table ===
  Model     Exact T1(%)    Exact T3(%)    Exact T5(%)      Base T1(%)     Base T3(%)     Base T5(%)
-------  --------------  -------------  -------------  --------------  -------------  -------------
 TypeT5           71.4%          77.2%          78.9%           78.1%          83.7%          85.4%
  +Ours  (+13.3%) 80.9%  (+7.4%) 82.9%  (+5.8%) 83.5%   (+9.1%) 85.2%  (+4.3%) 87.3%  (+3.0%) 88.0%
-------  --------------  -------------  -------------  --------------  -------------  -------------
  Tiger           67.8%          78.0%          80.2%           75.8%          85.5%          88.0%
  +Ours  (+15.2%) 78.1%  (+7.3%) 83.7%  (+6.0%) 85.0%  (+10.0%) 83.4%  (+4.9%) 89.7%  (+3.7%) 91.3%
-------  --------------  -------------  -------------  --------------  -------------  -------------
TypeGen           65.4%          73.4%          75.0%           71.6%          79.9%          81.6%
  +Ours  (+12.5%) 73.6%  (+7.6%) 79.0%  (+6.8%) 80.1%   (+9.9%) 78.7%  (+5.4%) 84.2%  (+4.9%) 85.6%

**Note:** The latest pyright update has introduced minor changes to the output.


=== Function Signature Table ===
... (continue with the rest of the results)
```

## Reproduction Guide

**Note: This process may take some time**

1. Download Benchmarks

To run the analysis, you need to prepare the data.
- [ManyTypes4Py](https://github.com/saltudelft/many-types-4-py-dataset)
- [BetterTypes4Py](https://github.com/utopia-group/TypeT5)

Due to the deletion of some original repositories, we provide the specific versions of the datasets used in our experiments via [Hugging Face](https://huggingface.co/datasets/marinelay/TypeCare-Datasets).

You are able to download the datasets via the following commands:

```bash
# in the Docker container
cd /home/TypeCare

# Download and extract datasets
hf download --repo-type dataset marinelay/TypeCare-Datasets --local-dir .
tar -I 'zstd -T8' -xvf BetterTypes4Py.tar.zst
tar -I 'zstd -T8' -xvf ManyTypes4Py.tar.zst
```

**Note:** Use -T8 to utilize 8 CPU cores for faster extraction; adjust as needed.

2. Run pre-analysis

**This section will create `data/` contents of the pre-computed result**

Generate static analysis results using pyright:

```bash
# in the Docker container
cd /home/TypeCare

# Run pyright for each candidates
python -m pre_analysis.run --tool=<typet5|tiger|typegen>
```
It makes the result of static type analysis in the directory `data`.


3. Run Re-ranking & Augmentation

**This section will create `output/` contents of the pre-computed result**

Now, you can refine the model answers using the trained type similarity model:
```bash
# Run for all tools
python -m analysis.run --all

# Or specify a single tool
python -m analysis.run --tool=<typet5|typegen|tiger>
```
Then, the analysis will be performed and the results will be saved in the `output` directory.

## Train the Type Similarity Model

**This section will create `predictions/typesim_model` contents of the pre-computed result**

If you want to train the type similarity model, please follow command:
```bash
python -m train_model.train.py
python -m train_model.train_many4types.py
```

## Citation

TODO:
If you find this work useful for your research, please consider citing our paper ...


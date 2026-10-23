# Leakage-Free Identification of Encrypted Web Services with a Screening Gateway

This repository contains reproducible experimental code for S2X, the system of the paper currently under review at The Web Conference (WWW) 2027.

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Environment Variables Configuration](#environment-variables-configuration)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

S2X identifies the Web service behind an encrypted session from the signed lengths of its first n = 30 packets only (no payload bytes, IP addresses, ports, SNI, or timing).
A lightweight gate scores every session against one learned (length × position) table per service and either finalizes the answer, rejects the session as an unseen service (open world), or defers it to a back-end classifier (PL-BERT).
S2X operates in three modes:
- **S2X-C**: Cascade mode (gate → PL-BERT)
- **S2X-G**: Gate-only mode
- **S2X-B**: PL-BERT-only mode

The gate table can be the **learned** table (default) or the **frequency** table it is initialized with.

## Environment Setup

### 1. Python Version
- Python 3.12.2

### 2. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following packages:

```
numpy==1.26.4
pandas==3.0.3
pyarrow==21.0.0
python-dotenv==1.2.1
scikit-learn==1.9.0
torch==2.6.0+cu124
transformers==4.45.2
```

**Note**: The PyTorch package in `requirements.txt` is built with CUDA 12.4. If you need a different CUDA version or CPU-only installation, modify the PyTorch package version accordingly.

### 3. PL-BERT Model

PL-BERT uses the BERT-base architecture (12 layers, hidden size 768, 12 heads) and is trained from scratch, so no pre-trained weights or tokenizer files are downloaded.

## Environment Variables Configuration

Create a `.env` file in the project root (`.env.example` is a complete template) and configure the following environment variables.

### Required Environment Variables

#### Basic Settings
```bash
DIM=30                     # prefix length n (first n packets)
LABEL=41                   # number of (known) services
MODE=s2x-c                 # execution mode: s2x-c, s2x-g, s2x-b
OPENWORLD=False            # open-world setting: True/False
IS_TRAIN=True              # True (training), False (inference)
DEVICE=cuda                # cuda or cpu
```

#### Data Paths
```bash
# Preprocessed data (parquet format)
TRAIN_DF=/path/to/preprocessed_train_df.parquet
VALID_DF=/path/to/preprocessed_valid_df.parquet
TEST_DF=/path/to/preprocessed_test_df.parquet
OPEN_TEST_DF=/path/to/preprocessed_openworld_test_df.parquet

# Raw data (before preprocessing)
RAW_TRAIN_DF=/path/to/train.parquet
RAW_VALID_DF=/path/to/valid.parquet
RAW_TEST_DF=/path/to/test.parquet

# Raw data column names
RAW_X_LABEL=SPLT-Data      # signed payload lengths
RAW_Y_LABEL=Label          # service label

# Output directory of the preprocessed data
DATASET_PATH=/path/to/dataset
```

#### Open-World Settings (service holdout)
```bash
HOLDOUT_RATIO=30           # percent of the services withheld as unseen (10, 20, 30, 40, 50 in the paper)
HOLDOUT_SEED=0             # holdout draw (0, 1, 2 in the paper)
```

#### Training Hyperparameters

**Gate (learned table)**
```bash
GATE_BATCH_SIZE=32
GATE_LR=0.001
GATE_EPOCH=10
```

**PL-BERT Model**
```bash
SEED=42
PL_BERT_BATCH_SIZE=32
PL_BERT_EPOCH=10
PL_BERT_LR=0.00005
PL_BERT_WARMUP_RATIO=0.1
```

#### Decision Rule
```bash
THETA0_PERCENTILE=5        # reject threshold θ0: 5th percentile of the gate confidence on validation data (open world)
THETA1_PERCENTILE=65       # finalize threshold θ1: 65th percentile of the confidence of misidentified validation sessions
THETA2=0.97                # PL-BERT rejection threshold θ2 (open world)
GATE_TABLE=learned         # learned or frequency
MEASURE_THROUGHPUT=False   # also report per-session throughput (flows/s)
```

#### Output and Model Paths
```bash
# Output directory (models are saved here during training)
OUTPUT_PATH=./outputs

# Trained models (required for inference)
GATE_MODEL_PATH=/path/to/gate_tables.npz
BERT_MODEL_PATH=/path/to/pl_bert
```

### Example .env File

```bash
# Basic settings
DIM=30
LABEL=41
MODE=s2x-c
OPENWORLD=False
IS_TRAIN=True
DEVICE=cuda

# Data paths
TRAIN_DF=./data/CipherSpectrum/preprocessed_train_df.parquet
VALID_DF=./data/CipherSpectrum/preprocessed_valid_df.parquet
TEST_DF=./data/CipherSpectrum/preprocessed_test_df.parquet
OPEN_TEST_DF=./data/CipherSpectrum/holdout_r30_seed0/preprocessed_openworld_test_df.parquet
DATASET_PATH=./data/CipherSpectrum

# Open world
HOLDOUT_RATIO=30
HOLDOUT_SEED=0

# Training hyperparameters
SEED=42
GATE_BATCH_SIZE=32
GATE_LR=0.001
GATE_EPOCH=10
PL_BERT_BATCH_SIZE=32
PL_BERT_EPOCH=10
PL_BERT_LR=0.00005
PL_BERT_WARMUP_RATIO=0.1

# Decision rule
THETA0_PERCENTILE=5
THETA1_PERCENTILE=65
THETA2=0.97
GATE_TABLE=learned

# Output paths
OUTPUT_PATH=./outputs
GATE_MODEL_PATH=./outputs/1234567890/gate_tables.npz
BERT_MODEL_PATH=./outputs/1234567890/pl_bert
```

## Data Preparation

### 1. Provided Data

`data/` contains the preprocessed closed-world splits of the two public TLS 1.3 datasets used in the paper:

| Directory | Dataset | Services | Train / Valid / Test |
|---|---|---|---|
| `data/CSTNET_TLS_1.3/` | CSTNET-TLS1.3 (transport payload lengths) | 120 | 37,097 / 4,638 / 4,637 |
| `data/CipherSpectrum/` | CipherSpectrum (Mix) | 41 | 32,964 / 4,100 / 4,141 |

Each `preprocessed_{train,valid,test}_df.parquet` has the columns `id`, `x` (signed packet lengths), and `y` (service id); `label_map.json` maps the original labels of the dataset to the ids.

### 2. Raw Data Format

Raw data should be in parquet format, one row per session. The column names **must exactly match** the values you set in the `.env` file:
- `{RAW_X_LABEL}`: signed transport-payload lengths of the packets (+ and − for the two directions; packets without payload have length 0)
- `{RAW_Y_LABEL}`: service label

### 3. Data Preprocessing

```bash
python main.py preprocess
```

This writes `preprocessed_{train,valid,test}_df.parquet` and `label_map.json` to `DATASET_PATH`. Each session is encoded as follows:
- lengths are capped at ±1500 bytes;
- the leading packets without payload (the TCP handshake) are dropped, so every session starts at its first packet with payload; later payload-less packets (pure ACKs) stay as length 0.

The gate represents the first n packets as a 3001 × n array (row = signed length + 1500); positions past the end of a short session are padded with length 0 (row 1500). PL-BERT reads the tokens signed length + 2500 and pads short sessions with `[PAD]`.

Service ids follow the sorted order of the training labels.

### 4. Open-World Data (Service Holdout)

The open world withholds a random `HOLDOUT_RATIO`% of the services of the same dataset from training and validation, and tests on the known test sessions mixed with an equal number of sessions of the withheld services (label `-1`). To create a holdout from the preprocessed closed-world splits (`TRAIN_DF`, `VALID_DF`, `TEST_DF`):

```bash
python main.py holdout
```

This writes `DATASET_PATH/holdout_r{ratio}_seed{seed}/` with `preprocessed_train_df.parquet`, `preprocessed_valid_df.parquet`, `preprocessed_openworld_test_df.parquet` (known services relabeled 0..k−1), and `holdout.json` (known and withheld service ids). Running `python main.py preprocess` with `HOLDOUT_RATIO` set also creates it.

## Usage

### 1. Training

Set `IS_TRAIN=True` in the `.env` file and run:

```bash
python main.py
```

Training process:
1. Build the frequency table
2. Train the learned table (initialized with the frequency table)
3. Train the PL-BERT model

Trained models are saved in the `OUTPUT_PATH/{timestamp}/` directory:
- `gate_tables.npz`: learned and frequency tables (n × 3001 × services)
- `pl_bert/`: PL-BERT model directory

For the open world, train on the holdout splits (`TRAIN_DF`, `VALID_DF` of `holdout_r{ratio}_seed{seed}/`) with `LABEL` set to the number of known services printed by `python main.py holdout`.

### 2. Inference

Set `IS_TRAIN=False` in the `.env` file and specify the trained model paths:

```bash
# Modify .env file
IS_TRAIN=False
GATE_MODEL_PATH=./outputs/1234567890/gate_tables.npz
BERT_MODEL_PATH=./outputs/1234567890/pl_bert
```

Then run:
```bash
python main.py
```

Inference process:
1. Load models
2. Set the gate thresholds θ1 (finalize) and θ0 (reject) on the validation data (`VALID_DF`)
3. Perform inference on the test data (`TEST_DF`, or `OPEN_TEST_DF` when `OPENWORLD=True`)

The result (Macro-F1; deferral rate for S2X-C; unknown recall and false rejection of known sessions in the open world) is printed as JSON. In the open world, Macro-F1 is computed over the known services plus *Unknown*.

### 3. Mode-Specific Behavior

#### s2x-c (Cascade)
- The gate scores the session against every service table and takes its confidence ρ* in the best service c*
- Open world: rejects the session as *Unknown* if ρ* < θ0
- Finalizes c* if ρ* ≥ θ1; otherwise defers the session to PL-BERT
- Open world: PL-BERT labels a deferred session *Unknown* if its maximum class probability is below θ2

#### s2x-g (Gate-only)
- Uses only the gate (rejection by θ0 in the open world)
- Fastest inference speed

#### s2x-b (PL-BERT-only)
- Uses only PL-BERT on every session (rejection by θ2 in the open world)

### 4. Throughput

With `MEASURE_THROUGHPUT=True`, inference also reports per-session throughput (flows/s) on the first 3,000 test sessions: each session is decided on its own, from its prefix already in memory, without batching. The paper pins every job to one CPU core with one thread (PyTorch uses one thread here); neural models additionally use the GPU:

```bash
taskset -c 0 python main.py
```

## Troubleshooting

### Model Loading Errors
- Verify model paths are correct
- Check that model files exist
- Verify `IS_TRAIN` setting is correct
- In the open world, `LABEL` must be the number of known services of the holdout

### Out of Memory
- Reduce `PL_BERT_BATCH_SIZE` and `GATE_BATCH_SIZE`
- Change to `DEVICE=cpu` to use CPU

### Preprocessing Errors
- Verify parquet file paths
- Check data format: column names in the raw data must match the values specified in `.env` file (`RAW_X_LABEL` and `RAW_Y_LABEL`)
- Every label of the validation and test data must also appear in the training data

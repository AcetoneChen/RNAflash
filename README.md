# RNAflash: Fast and Sensitive Structure-aware RNA Homology Detection and Alignment

RNAflash retrieves related RNA sequences with learned embeddings and FAISS, then
builds multiple sequence alignments (MSAs). This repository provides inference
entry points and database preparation tools.

| Script | Purpose |
| --- | --- |
| `embed_database_indexed.py` | Encode a FASTA database and build its FAISS index. |
| `search_faiss_mmp.py` | Retrieve the top-k database sequences for each query. |
| `search_faiss_prewarm.py` | Optionally warm the index and FASTA page cache before retrieval. |
| `align.py` | Rerank and align retrieved sequences with RNAflashA. |
| `RNAflashL.py` | Run FAISS retrieval followed by nhmmer-based MSA construction. |

## Installation

Create the Conda environment on Linux x86_64:

```bash
conda env create -f requirements.yml
conda activate RNAflash
```

Installation typically takes about 30 minutes, depending on network speed and
package caches, excluding model and database downloads.

Dependencies are pinned in `requirements.yml`, including PyTorch 2.6.0 (CUDA
12.4), FAISS GPU 1.13.2 (CUDA 12.6), and HMMER 3.4 (`nhmmer` and
`esl-reformat`). An NVIDIA GPU and a driver compatible with CUDA 12.6 are
required. Tested on Ubuntu 16.04.6 LTS with a single RTX 3090 (24 GiB).

Download the [checkpoints](https://drive.google.com/file/d/103fUAx3c10UUAdVR7fczqBBUNQDW09y9/view?usp=sharing)
and place them at `ckpts/RNAflashR.ckpt` (retrieval) and `ckpts/RNAflashA.ckpt`
(RNAflashA alignment). Run all commands from the project root.

## Prepare the database

Download all 43 database parts (`RNAcentral_27.tar.gz.part000` through `part042`)
from [Zenodo](https://zenodo.org/records/23492657) into the project root, then
merge and extract:

```bash
mkdir -p datasets
LC_ALL=C cat RNAcentral_27.tar.gz.part[0-9][0-9][0-9] > RNAcentral_27.tar.gz &&
tar -xzvf RNAcentral_27.tar.gz -C datasets/
```

The expected database files are:

```text
datasets/RNAcentral_27/
├── rnacentral_active_1022_v27.fasta
├── rnacentral_active_1022_v27_idx.npy
└── faiss/
    └── faiss_index_fp32.index
```

For benchmark queries, download `benchmark_dataset.zip` from
[Zenodo](https://zenodo.org/records/23599961) and extract it into `datasets/`
from the project root:

```bash
mkdir -p datasets
unzip /path/to/benchmark_dataset.zip -d datasets/
```

Alternatively, encode your own FASTA database. Use one header line and one
unwrapped sequence line per record. Set `FASTA_PATH`, `FASTA_INDEX_PATH`,
`SAVE_DIR`, and `device` (`cuda:0` for a single GPU) in
`embed_database_indexed.py`, then run:

```bash
python embed_database_indexed.py
```

Use a fresh output directory for a different database, and adjust `NLIST` and
`TRAIN_SIZE` for smaller databases. Encoding uses up to 1,022 nucleotides per
sequence. Update `INDEX`, `FASTA_IDX`, and `DATABASE_FASTA` in the retrieval
scripts to match the generated files. The FASTA, offset map, and FAISS index must
come from the same database build.

## Run inference

Place one query per file in `datasets/<target>/fasta/<query_id>.fasta`, using one
unwrapped sequence line. Replace `my_queries` below with your target directory
name (e.g. `RNAsolo_singlechain` as a demo).

For FAISS retrieval followed by nhmmer alignment:

```bash
python RNAflashL.py --target my_queries --device cuda:0
```

Defaults: `--probe 512`, `--topk 5000`, and prewarming disabled. Add `--prewarm`
to enable it. It is highly recommended to use prewarm, as it reads faiss database to cache and substantially reduce search time. Use `--save-dir /path/to/output` to change the output base directory.
MSAs are saved to:

```text
datasets/my_queries/a2m_RNAflashL_nprobe512_top5000/<query_id>.a2m
```

Each query produces an `.a2m` file containing the query sequence and aligned
database hits.

For retrieval followed by RNAflashA alignment:

```bash
python search_faiss_mmp.py --target my_queries --probe 512 --topk 1000 --device cuda:0
python align.py --target my_queries --probe 512 --topk 1000 --device cuda:0
```

Use the same `--target`, `--probe`, and `--topk` for search and alignment.
Both scripts default to `RNAsolo_singlechain`, 512, and 1000, respectively,
and use `cuda:0`. Alignment reads from
`datasets/my_queries/search_all/search_nprobe512_top1000` and writes to
`datasets/my_queries/a2m_RNAflashA_nprobe512_top1000` for the example above.
Override alignment paths with `--fasta-dir` and `--output-dir`, select a model
with `--checkpoint`, or adjust `--batch-size` (default: 8).

Standalone prewarming is optional but recommended: set `INDEX` and `DATABASE_FASTA` in
`search_faiss_prewarm.py` to the same paths used by retrieval, then run
`python search_faiss_prewarm.py` before the search.

Reference runtimes are shown in our paper. For the full `RNAsolo_singlechain`
set on a single RTX 3090, the FAISS search pipeline takes tens of seconds,
prewarming and RNAflash-L each take a few minutes, and RNAflashA alignment
takes about half an hour.

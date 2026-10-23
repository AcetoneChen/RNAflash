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

`requirements.yml` pins the core dependencies from the development environment:
PyTorch 2.6.0 (CUDA 12.4), Lightning 2.6.0, FAISS GPU 1.13.2 (CUDA 12.6), and
HMMER 3.4, including `esl-reformat`. Use an NVIDIA driver compatible with CUDA 12.6.
Alternative backbones may require additional packages, such as FlashAttention for RiNALMo.

The five scripts require the project's `src/` package and trained checkpoints;
they are not standalone model implementations. Place the retrieval checkpoint at
`ckpts/RNAflashR.ckpt`, and the alignment checkpoint at `ckpts/RNAflashA.ckpt`
when using `align.py`. The checkpoints can be downloaded [here](https://drive.google.com/file/d/103fUAx3c10UUAdVR7fczqBBUNQDW09y9/view?usp=sharing).

## Prepare the database

Database files are not included in this repository. Create a `dataset/` directory,
download the database archive from Zenodo (coming soon), and extract it there.


The current scripts expect `datasets/` (plural). For a fresh checkout, create a
compatibility symlink:

```bash
mkdir -p dataset
ln -s dataset datasets
tar -xzvf /path/to/RNAcentral_27.tar.gz -C dataset/
```

If the Zenodo download is split into numbered files, download all parts and the
`RNAcentral_27.tar.gz.parts.json` manifest into one directory. In that directory,
merge and verify them before extracting the resulting archive into `dataset/`:

```bash
cat RNAcentral_27.tar.gz.part[0-9][0-9][0-9] > RNAcentral_27.tar.gz
python -c 'import json; m=json.load(open("RNAcentral_27.tar.gz.parts.json")); print(m["md5"] + "  RNAcentral_27.tar.gz")' | md5sum -c -
```

Proceed with extraction only if the checksum reports `OK`.

If `datasets/` already exists, extract into that directory instead of creating the
symlink. The expected database files are:

```text
datasets/RNAcentral_27/
├── rnacentral_active_1022_v27.fasta
├── rnacentral_active_1022_v27_idx.npy
└── faiss/
    └── faiss_index_fp32.index
```

Alternatively, encode your own FASTA database. Use one header line and one
unwrapped sequence line per record. Set `FASTA_PATH`, `FASTA_INDEX_PATH`,
`SAVE_DIR`, and the CUDA device in `embed_database_indexed.py`, then run:

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
unwrapped sequence line. Replace `my_queries` below with your target directory name.

For FAISS retrieval followed by nhmmer alignment:

```bash
python RNAflashL.py --target my_queries --device cuda:0
```

Defaults: `--probe 512`, `--topk 5000`, and prewarming disabled. Add `--prewarm`
to enable it, or `--save-dir /path/to/output` to change the output base directory.
MSAs are saved to:

```text
datasets/my_queries/a2m_RNAflashL_nprobe512_top5000_exp/<query_id>.a2m
```

For retrieval followed by RNAflashA alignment:

```bash
python search_faiss_mmp.py --target my_queries --probe 512 --topk 5000
python align.py
```

Before running, select an available CUDA device in both scripts and set
`align.py`'s `fasta_dir` to
`datasets/my_queries/search_all/search_nprobe512_top5000`; set `a2m_dir` to your
desired output directory. These settings are configured inside `align.py`.

Standalone prewarming is optional: set `INDEX` and `DATABASE_FASTA` in
`search_faiss_prewarm.py` to the same paths used by retrieval, then run
`python search_faiss_prewarm.py` before the search.

#!/usr/bin/env python3
"""RNAflashL: optional prewarm -> FAISS top-k retrieval -> nhmmer MSA.

Usage:
    python RNAflashL.py
    python RNAflashL.py --prewarm --probe 512 --topk 5000 --save-dir ./outputs

All pipeline steps are implemented here; no other pipeline script is imported
or executed. The trained model and tokenizers still use the project's src package.
Relative paths are resolved from this file's directory, as in RNAflashL.sh.
"""
import argparse
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
INDEX = Path("datasets/RNAcentral_27/faiss/faiss_index_fp32.index")
FASTA_IDX = Path("datasets/RNAcentral_27/rnacentral_active_1022_v27_idx.npy")
DATABASE_FASTA = Path("datasets/RNAcentral_27/rnacentral_active_1022_v27.fasta")
CHECKPOINT = Path("ckpts/RNAflashR.ckpt")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", "--nprob", dest="nprob", type=positive_int, default=512)
    parser.add_argument("--topk", type=positive_int, default=5000)
    parser.add_argument("--target", default="RNAsolo_singlechain")
    parser.add_argument("--save-dir", "--save_dir", type=str,
                        help="output base directory (default: datasets/<target>/)")
    parser.add_argument("--prewarm", action="store_true",
                        help="read the FAISS index and database FASTA into page cache (default: off)")
    parser.add_argument("--device", default="cuda:0", help="embedding device (default: cuda:0)")
    args = parser.parse_args(argv)
    if args.save_dir is None:
        args.save_dir = f"datasets/{args.target}/"
    return args


def prewarm(paths):
    """Sequentially read the exact files used by retrieval, in bounded chunks."""
    for path in paths:
        nbytes = 0
        with path.open("rb", buffering=0) as handle:
            while chunk := handle.read(8 << 20):
                nbytes += len(chunk)
        print(f"[prewarm] {path}: {nbytes / 2**30:.2f} GiB", flush=True)


class DatabaseConverter:
    def __init__(self, model_type, alphabet):
        if model_type == "fm" and alphabet is not None:
            self.seqconverter = alphabet.get_batch_converter()
            self.pad_idx = alphabet.padding_idx
        elif model_type == "rinalmo":
            from src.data.converters import ri_converter
            self.seqconverter = ri_converter(alphabet)
            self.pad_idx = alphabet.pad_idx
        elif model_type == "ernie_rna":
            from src.models.ernierna.converter import ErnieRNA_Converter
            self.seqconverter = ErnieRNA_Converter()
            self.pad_idx = 1
        else:
            raise NotImplementedError(f"Unsupported backbone: {model_type}")

    def __call__(self, batch):
        pairs = [(name, seq[:1022]) for name, seq in batch]
        _, _, tokens = self.seqconverter(pairs)
        return tokens, tokens == self.pad_idx


def embed_queries(sequences, device_name):
    import lightning as L
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from src.models.RNAflashR import RNAflashR

    L.seed_everything()
    device = torch.device(device_name)
    model = RNAflashR.load_from_checkpoint(str(CHECKPOINT))
    model.to(device)
    model.eval()
    converter = DatabaseConverter(model.backbone_k.pretrained_model, model.backbone_k.alphabet)
    loader = DataLoader([("", seq) for seq in sequences], batch_size=16,
                        collate_fn=converter, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for tokens, mask in tqdm(loader, desc="Embedding queries",
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            embeddings.append(model.embed(tokens.to(device), mask.to(device)).detach().cpu())
    result = np.ascontiguousarray(torch.vstack(embeddings).numpy(), dtype=np.float32)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def faiss_search(query_files, sequences, embeddings, retrieved_dir, nprob, topk):
    import faiss
    import numpy as np

    index = faiss.read_index(str(INDEX), faiss.IO_FLAG_MMAP)
    index.nprobe = nprob
    idx_map = np.load(FASTA_IDX)
    scores, indices = index.search(embeddings, topk)
    retrieved_dir.mkdir(parents=True, exist_ok=True)
    with DATABASE_FASTA.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as database:
            for i, query_path in enumerate(query_files):
                with (retrieved_dir / query_path.name).open("w") as output:
                    output.write(f">query\n{sequences[i]}\n")
                    for score, idx in zip(scores[i], indices[i]):
                        # FAISS pads with -1 when fewer than topk neighbors are found.
                        if idx < 0:
                            continue
                        start, end = idx_map[idx]
                        lines = database[int(start):int(end) + 1].decode("utf-8").splitlines()
                        description = lines[0]
                        sequence = "".join(lines[1:]).upper().replace("T", "U")
                        output.write(f"{description} | score: {score}\n{sequence}\n")


def run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def build_msa(query_path, query_seq, retrieved_fasta, output_path):
    from Bio import SeqIO

    with tempfile.TemporaryDirectory(prefix="RNAflashL_") as tmp:
        tmp = Path(tmp)
        database_path = tmp / "targets.fasta"
        sto_path = tmp / "hits.sto"
        a2m_path = tmp / "hits.a2m"
        profile_path = tmp / "query.hmm"

        # The leading query is not part of the nhmmer search database.
        with retrieved_fasta.open() as handle:
            records = SeqIO.parse(handle, "fasta")
            first = next(records, None)
            if first is None or first.id != "query":
                raise ValueError(f"{retrieved_fasta}: FAISS output must start with >query")
            hit_count = SeqIO.write(records, database_path, "fasta")

        alignment = ""
        if hit_count:
            run(["nhmmer", "-A", sto_path, "--rna", query_path, database_path])
            if sto_path.stat().st_size:
                reformat = ["esl-reformat", "-o", a2m_path, "--replace",
                            "acgturyswkmbdhvn:................", "-r", "a2m", sto_path]
                run(reformat)
                with a2m_path.open() as handle:
                    first_hit = next(SeqIO.parse(handle, "fasta"))
                if len(first_hit.seq) < len(query_seq):
                    run(["hmmbuild", "--rna", profile_path, query_path])
                    run(["hmmalign", "--rna", "-o", sto_path, "--informat", "a2m",
                         profile_path, a2m_path])
                    run(reformat)
                with a2m_path.open() as handle:
                    if any(len(record.seq) != len(query_seq)
                           for record in SeqIO.parse(handle, "fasta")):
                        raise ValueError(f"{query_path.stem}: MSA length differs from query length")
                alignment = a2m_path.read_text()

        # No significant hits still produces a valid query-only A2M.
        output_path.write_text(f">query\n{query_seq}\n" + alignment)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    from Bio import SeqIO

    query_dir = Path("datasets") / args.target / "fasta"
    query_files = sorted(query_dir.glob("*.fasta"))
    if not query_files:
        raise SystemExit(f"No query FASTA files found in {query_dir}")
    for tool in ("nhmmer", "esl-reformat", "hmmbuild", "hmmalign"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Required executable not found in PATH: {tool}")
    for path in (CHECKPOINT, INDEX, FASTA_IDX, DATABASE_FASTA):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    sequences = [str(SeqIO.read(path, "fasta").seq).upper().replace("T", "U")
                 for path in query_files]

    output_dir = Path(args.save_dir) / f"a2m_RNAflashL_nprobe{args.nprob}_top{args.topk}"
    retrieved_dir = (Path("datasets") / args.target / "search_all"
                     / f"search_nprobe{args.nprob}_top{args.topk}")

    if args.prewarm:
        prewarm((INDEX, DATABASE_FASTA))
    print(f"[1/2] FAISS search: nprobe={args.nprob}, topk={args.topk}", flush=True)
    embeddings = embed_queries(sequences, args.device)
    faiss_search(query_files, sequences, embeddings, retrieved_dir, args.nprob, args.topk)
    print(f"[2/2] Build nhmmer MSAs: {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (query_path, query_seq) in enumerate(zip(query_files, sequences), 1):
        build_msa(query_path, query_seq, retrieved_dir / query_path.name,
                  output_dir / f"{query_path.stem}.a2m")
        print(f"[{i}/{len(query_files)}] {query_path.stem}", flush=True)
    print(f"Saved MSAs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

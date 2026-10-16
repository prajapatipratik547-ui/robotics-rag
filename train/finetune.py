"""Step 3 of fine-tuning: train bge-small on the filtered synthetic pairs and
export it to ONNX for the pipeline (FastEmbed loads it via DENSE_MODEL_PATH).

Loss: multiple-negatives ranking (each query's own chunk is the positive, the
other chunks in the batch are negatives), with batches that never hold the
same chunk twice. Hyperparameters are fixed in advance, not tuned on the test
set; the number of epochs is picked on a validation split of the synthetic
queries (5% of chunks held out), never on the 85 test questions.

    python train/finetune.py --name bge-small-robotics                     # all documents
    python train/finetune.py --name bge-small-robotics-noA --exclude-fold A
    python train/finetune.py --name bge-small-base --no-train              # untrained export: the fair baseline

Runs in the training environment (requirements-train.txt).
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import numpy as np
import onnxruntime
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import AutoModel, AutoTokenizer

from folds import CHUNKS_PATH, document_folds, load_jsonl
from src.passages import embedding_input

BASE_MODEL = "BAAI/bge-small-en-v1.5"
PAIRS_PATH = ROOT / "train" / "data" / "pairs.jsonl"
# Models (~130 MB each) and checkpoints stay out of the project folder, which
# is synced to OneDrive here. Point DENSE_MODEL_PATH at MODELS_DIR/<name>.
MODELS_DIR = Path.home() / "venvs" / "robotics-rag-models"
CHECKPOINT_DIR = Path.home() / "venvs" / "robotics-rag-checkpoints"
# FastEmbed's own copy of bge-small: its tokenizer and config files are copied
# next to the exported model so FastEmbed reads them exactly as it reads its own.
FASTEMBED_SNAPSHOTS = ROOT / ".fastembed_cache" / "models--Qdrant--bge-small-en-v1.5-onnx-Q" / "snapshots"
VALIDATION_SHARE = 0.05
CHECKPOINT_STEPS = 32


def is_validation(chunk_id: str) -> bool:
    """A stable 5% of chunks, chosen by hash, whose queries are held out."""
    return int(hashlib.sha1(chunk_id.encode()).hexdigest(), 16) % 100 < VALIDATION_SHARE * 100


def load_data(exclude_fold: str | None, pairs_path: Path = PAIRS_PATH) -> tuple[Dataset, InformationRetrievalEvaluator]:
    folds = document_folds()
    chunks = {c["chunk_id"]: c for c in load_jsonl(CHUNKS_PATH)}
    allowed = {cid for cid, c in chunks.items() if folds[c["source_file"]] != exclude_fold}
    pairs = [p for p in load_jsonl(pairs_path) if p["verdict"] == "kept" and p["chunk_id"] in allowed]
    train = [p for p in pairs if not is_validation(p["chunk_id"])]
    val = [p for p in pairs if is_validation(p["chunk_id"])]
    print(f"{len(allowed)} chunks allowed, {len(train)} training pairs, {len(val)} validation queries")

    dataset = Dataset.from_dict({
        "anchor": [p["query"] for p in train],
        "positive": [embedding_input(chunks[p["chunk_id"]]) for p in train],
    })
    evaluator = InformationRetrievalEvaluator(
        queries={str(i): p["query"] for i, p in enumerate(val)},
        corpus={cid: embedding_input(chunks[cid]) for cid in allowed},
        relevant_docs={str(i): {p["chunk_id"]} for i, p in enumerate(val)},
        name="val", accuracy_at_k=[1, 5], precision_recall_at_k=[5], mrr_at_k=[5], ndcg_at_k=[10], map_at_k=[10],
    )
    return dataset, evaluator


def train(model: SentenceTransformer, dataset: Dataset, evaluator, name: str, args) -> None:
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)  # batch of 32 at the memory cost of 8
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR / name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # a repeated chunk would be a false negative
        # Validate and checkpoint every ~half epoch (~15 min on a laptop CPU), so an
        # interrupted run resumes instead of starting over.
        eval_strategy="steps",
        eval_steps=CHECKPOINT_STEPS,
        save_strategy="steps",
        save_steps=CHECKPOINT_STEPS,
        save_total_limit=2,  # the best and the latest
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_mrr@5",
        logging_steps=10,
        report_to="none",
        seed=0,
        use_cpu=True,
    )
    resume = any(CHECKPOINT_DIR.joinpath(name).glob("checkpoint-*"))
    if resume:
        print("resuming from the last checkpoint")
    SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=dataset, loss=loss, evaluator=evaluator
    ).train(resume_from_checkpoint=resume or None)


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (reverts when it exits)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


class LastHiddenState(torch.nn.Module):
    """BertModel returning only last_hidden_state, the output FastEmbed pools."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]


def export_onnx(model_dir: Path, out_dir: Path) -> None:
    model = LastHiddenState(AutoModel.from_pretrained(model_dir).eval())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = tokenizer(["a sample passage", "another one"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(
        model, tuple(sample[n] for n in names), str(out_dir / "model.onnx"),
        input_names=names, output_names=["last_hidden_state"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in [*names, "last_hidden_state"]},
        opset_version=17, dynamo=False,
    )
    snapshot = next(FASTEMBED_SNAPSHOTS.iterdir())
    for f in snapshot.iterdir():
        if f.suffix != ".onnx":
            shutil.copyfile(f, out_dir / f.name)


def check_parity(st_model: SentenceTransformer, out_dir: Path) -> float:
    """Max difference between the ONNX export (CLS + normalize, as FastEmbed
    does) and the trained model's own embeddings, on real passages."""
    texts = [embedding_input(c) for c in load_jsonl(CHUNKS_PATH)[:: 150]]
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    session = onnxruntime.InferenceSession(str(out_dir / "model.onnx"))
    hidden = session.run(None, {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")})[0]
    onnx_vectors = hidden[:, 0] / np.linalg.norm(hidden[:, 0], axis=1, keepdims=True)
    return float(np.abs(onnx_vectors - st_model.encode(texts, normalize_embeddings=True)).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="output folder under models/")
    parser.add_argument("--exclude-fold", choices=["A", "B"], help="leave this fold's documents out of training")
    parser.add_argument("--no-train", action="store_true", help="export the base model unchanged")
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH, help="filtered pairs from filter_queries.py")
    parser.add_argument("--epochs", type=int, default=3, help="maximum; the best on validation is kept")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    torch.manual_seed(0)
    keep_awake()

    out_dir = MODELS_DIR / args.name
    st_dir = CHECKPOINT_DIR / args.name / "final"  # PyTorch copy; only the ONNX export is used
    model = SentenceTransformer(BASE_MODEL, device="cpu")
    model.max_seq_length = 512  # FastEmbed's limit; chunks are ~430 tokens with their prefix

    if not args.no_train:
        dataset, evaluator = load_data(args.exclude_fold, args.pairs)
        print("validation before:", {k: round(v, 3) for k, v in evaluator(model).items() if "mrr" in k or "accuracy@5" in k})
        train(model, dataset, evaluator, args.name, args)
        print("validation after:", {k: round(v, 3) for k, v in evaluator(model).items() if "mrr" in k or "accuracy@5" in k})

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(st_dir))
    export_onnx(st_dir, out_dir)
    print(f"exported {out_dir / 'model.onnx'}; ONNX vs PyTorch max difference: {check_parity(model, out_dir):.2e}")


if __name__ == "__main__":
    main()

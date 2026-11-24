import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bayes_inference import BayesianImputer
from language_models import BigramLanguageModel


def load_unigram_cdf(csv_path: Path, candidates: list[str]) -> np.ndarray:
    """Build a fast cumulative distribution over `candidates` from unigram_freq.csv."""
    counts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    counts[row[0].strip().lower()] = float(row[1])
                except ValueError:
                    continue

    weights = np.array([counts.get(w, 1.0) for w in candidates], dtype=np.float64)
    probs = weights / weights.sum()
    return np.cumsum(probs)


def sample_impute(
    imputer: BayesianImputer,
    w_prev: str,
    w_next: str,
    n_samples: int,
    rng: np.random.Generator,
    cdf: np.ndarray | None = None,
    method: str = "kn",
) -> str:
    """Approximate MAP inference by sampling N candidates and scoring unique draws."""
    n_vocab = len(imputer.candidates)
    if cdf is None:
        # Uniform proposal over V
        idxs = rng.integers(0, n_vocab, size=n_samples)
    else:
        # Importance sampling proposal from unigram distribution
        u = rng.random(size=n_samples)
        idxs = np.searchsorted(cdf, u)
        idxs = np.clip(idxs, 0, n_vocab - 1)

    unique_idxs = np.unique(idxs)
    best_word = imputer.candidates[unique_idxs[0]]
    best_score = float("-inf")

    for idx in unique_idxs:
        w = imputer.candidates[idx]
        score = imputer._transition_log_prob(w, w_prev, method=method)
        score += imputer._transition_log_prob(w_next, w, method=method)
        if score > best_score:
            best_score = score
            best_word = w

    return best_word


def extract_oracle_targets(data: list[dict]) -> list[tuple[str, str, str]]:
    """Return list of (w_prev, target_word, w_next) from ground-truth utterances."""
    targets = []
    for utt in data:
        gt_words = [w.lower() for w in utt["ground_truth"]]
        target_word = utt["target_word"].lower()

        idx = utt.get("target_idx")
        if idx is None or idx >= len(gt_words) or gt_words[idx] != target_word:
            if target_word in gt_words:
                idx = gt_words.index(target_word)
            else:
                continue

        w_prev = gt_words[idx - 1] if idx > 0 else "<s>"
        w_next = gt_words[idx + 1] if idx + 1 < len(gt_words) else "</s>"
        targets.append((w_prev, target_word, w_next))
    return targets


def save_plot(exact_ms: float, exact_acc: float, results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = sorted({r["n_samples"] for r in results})

    imp_rows = [r for r in results if r["sampler"] == "importance"]
    uni_rows = [r for r in results if r["sampler"] == "uniform"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left panel: Agreement with Exact Inference
    ax1.plot(
        sample_sizes,
        [r["exact_agreement"] * 100 for r in imp_rows],
        marker="o",
        label="Importance Sampling (Unigram)",
    )
    ax1.plot(
        sample_sizes,
        [r["exact_agreement"] * 100 for r in uni_rows],
        marker="s",
        label="Uniform Sampling",
    )
    ax1.axhline(100.0, linestyle="--", color="gray", alpha=0.7, label="Exact Inference (|V|=30k)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Samples (N)")
    ax1.set_ylabel("Agreement with Exact MAP (%)")
    ax1.set_title("Approximate vs. Exact Inference Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right panel: Latency (ms per utterance)
    ax2.plot(
        sample_sizes,
        [r["avg_ms"] for r in imp_rows],
        marker="o",
        label="Importance Sampling (Unigram)",
    )
    ax2.plot(
        sample_sizes,
        [r["avg_ms"] for r in uni_rows],
        marker="s",
        label="Uniform Sampling",
    )
    ax2.axhline(exact_ms, linestyle="--", color="gray", alpha=0.7, label=f"Exact ({exact_ms:.1f} ms)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Samples (N)")
    ax2.set_ylabel("Avg Latency per Blank (ms)")
    ax2.set_title("Inference Speed vs. Sample Budget")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "approx_vs_exact.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="Number of oracle targets (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading language model...")
    lm = BigramLanguageModel(vocab_size=30_000)
    lm.build_vocab(str(root / "data/unigram_freq.csv"), str(root / "data/phase1_outputs.json"))
    lm.fit(str(root / "data/sentences.txt"))
    imputer = BayesianImputer(lm)

    cdf = load_unigram_cdf(root / "data/unigram_freq.csv", imputer.candidates)

    with open(root / "data/phase1_outputs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    targets = extract_oracle_targets(data)
    if args.limit > 0:
        targets = targets[: args.limit]
    print(f"Running Exact vs. Approximate Inference on {len(targets)} oracle targets...")

    # 1. Exact Inference Baseline
    exact_preds = []
    exact_hits = 0
    t0 = time.perf_counter()
    for w_prev, target_word, w_next in tqdm(targets, desc="Exact (|V|=30k)"):
        top_preds = imputer.impute_blank(
            w_prev=w_prev, w_next=w_next, method="kn", use_bidirectional=True, top_k=1
        )
        pred = top_preds[0][0]
        exact_preds.append(pred)
        if pred == target_word:
            exact_hits += 1
    exact_total_s = time.perf_counter() - t0
    exact_ms = (exact_total_s / len(targets)) * 1000.0
    exact_acc = exact_hits / len(targets)
    print(f"Exact Inference | Top-1 Acc: {exact_acc:.4f} | Avg Latency: {exact_ms:.2f} ms")

    # 2. Approximate Inference Sweep
    sample_sizes = [100, 500, 1000, 2500, 5000, 10000]
    rng = np.random.default_rng(args.seed)
    approx_rows = []

    for sampler in ["importance", "uniform"]:
        use_cdf = cdf if sampler == "importance" else None
        for n in sample_sizes:
            agreements = 0
            hits = 0
            t0 = time.perf_counter()
            for (w_prev, target_word, w_next), exact_w in zip(targets, exact_preds):
                pred = sample_impute(imputer, w_prev, w_next, n, rng, cdf=use_cdf, method="kn")
                if pred == exact_w:
                    agreements += 1
                if pred == target_word:
                    hits += 1
            elapsed_ms = ((time.perf_counter() - t0) / len(targets)) * 1000.0
            row = {
                "sampler": sampler,
                "n_samples": n,
                "exact_agreement": round(agreements / len(targets), 4),
                "oracle_top1_acc": round(hits / len(targets), 4),
                "avg_ms": round(elapsed_ms, 2),
                "speedup": round(exact_ms / elapsed_ms, 2) if elapsed_ms > 0 else 0.0,
            }
            approx_rows.append(row)
            print(row)

    payload = {
        "num_targets": len(targets),
        "exact_top1_acc": round(exact_acc, 4),
        "exact_avg_ms": round(exact_ms, 2),
        "approx_results": approx_rows,
    }
    with open(out_dir / "approx_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_plot(exact_ms, exact_acc, approx_rows, out_dir)
    print(f"\nSaved results/approx_results.json and results/approx_vs_exact.png")


if __name__ == "__main__":
    main()
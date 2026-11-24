import argparse
import json
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayes_inference import BayesianImputer
from language_models import BigramLanguageModel, tokenize


def normalize_sentence(words: list[str]) -> str:
    return " ".join(w for w in words if w)


def extract_whisper_tokens(whisper_words: list[dict]) -> list[tuple[str, float]]:
    out = []
    for item in whisper_words:
        toks = tokenize(item["word"])
        w = toks[0] if toks else ""
        out.append((w, item["probability"]))
    return out


def eval_threshold_sweep(data: list[dict], noise_mode: str = "noise_-5db", taus=(0.2, 0.4, 0.6, 0.75, 0.9)):
    """Measure Precision/Recall/F1 of flagging actual Whisper errors across tau."""
    rows = []
    for tau in taus:
        tp = fp = fn = 0
        for utt in data:
            gt_set = set(utt["ground_truth"])
            tokens = extract_whisper_tokens(utt["runs"][noise_mode])

            for w, prob in tokens:
                is_error = (w not in gt_set)
                flagged = (prob < tau)
                if flagged and is_error:
                    tp += 1
                elif flagged and not is_error:
                    fp += 1
                elif not flagged and is_error:
                    fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({"tau": tau, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return rows


def evaluate_config(
    imputer: BayesianImputer,
    data: list[dict],
    noise_mode: str,
    tau: float,
    method: str,
    use_bidirectional: bool,
    acoustic_weight: float,
) -> dict:
    """Compute Raw WER, Imputed WER, and Target Recovery (Top-1 / Top-5) on flagged targets."""
    gt_texts = []
    raw_texts = []
    imp_texts = []

    flagged_total = 0
    top1_hits = 0
    top5_hits = 0

    for utt in tqdm(data, desc=f"{noise_mode} | {method} | bi={use_bidirectional} | aw={acoustic_weight}", leave=False):
        gt_words = utt["ground_truth"]
        target_word = utt["target_word"]
        whisper_run = utt["runs"][noise_mode]

        raw_tokens = [w for w, _ in extract_whisper_tokens(whisper_run)]
        resolved, imps = imputer.resolve_utterance(
            whisper_run,
            tau=tau,
            method=method,
            use_bidirectional=use_bidirectional,
            acoustic_weight=acoustic_weight,
        )

        gt_texts.append(normalize_sentence(gt_words))
        raw_texts.append(normalize_sentence(raw_tokens))
        imp_texts.append(normalize_sentence(resolved))

        for imp in imps:
            if imp["original_whisper"] != target_word and target_word not in raw_tokens:
                flagged_total += 1
                if imp["predicted"] == target_word:
                    top1_hits += 1
                if target_word in imp["top_5"]:
                    top5_hits += 1

    raw_wer = jiwer.wer(gt_texts, raw_texts)
    imp_wer = jiwer.wer(gt_texts, imp_texts)
    top1_acc = (top1_hits / flagged_total) if flagged_total > 0 else 0.0
    top5_acc = (top5_hits / flagged_total) if flagged_total > 0 else 0.0

    return {
        "noise_mode": noise_mode,
        "method": method,
        "use_bidirectional": use_bidirectional,
        "acoustic_weight": acoustic_weight,
        "raw_wer": round(raw_wer, 4),
        "imputed_wer": round(imp_wer, 4),
        "wer_delta": round(imp_wer - raw_wer, 4),
        "flagged_errors": flagged_total,
        "top1_acc": round(top1_acc, 4),
        "top5_acc": round(top5_acc, 4),
    }


def eval_oracle_masking(
    imputer: BayesianImputer,
    data: list[dict],
    method: str,
    use_bidirectional: bool,
) -> dict:
    """Mask the exact ground-truth target word and evaluate Phase 2 BN inference in isolation."""
    total = 0
    top1_hits = 0
    top5_hits = 0

    label = f"oracle | {method} | bi={use_bidirectional}"
    for utt in tqdm(data, desc=label, leave=False):
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

        top_preds = imputer.impute_blank(
            w_prev=w_prev,
            w_next=w_next,
            noisy_word="",
            whisper_prob=0.0,
            method=method,
            use_bidirectional=use_bidirectional,
            acoustic_weight=0.0,
            top_k=5,
        )
        pred_words = [w for w, _ in top_preds]

        total += 1
        if pred_words[0] == target_word:
            top1_hits += 1
        if target_word in pred_words:
            top5_hits += 1

    return {
        "method": method,
        "use_bidirectional": use_bidirectional,
        "total_targets": total,
        "top1_acc": round(top1_hits / total, 4) if total > 0 else 0.0,
        "top5_acc": round(top5_hits / total, 4) if total > 0 else 0.0,
    }


def save_plots(
    threshold_rows: list[dict],
    noise_rows: list[dict],
    oracle_rows: list[dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Threshold Calibration Curve
    taus = [r["tau"] for r in threshold_rows]
    plt.figure(figsize=(6, 4))
    plt.plot(taus, [r["precision"] for r in threshold_rows], marker="o", label="Precision")
    plt.plot(taus, [r["recall"] for r in threshold_rows], marker="s", label="Recall")
    plt.plot(taus, [r["f1"] for r in threshold_rows], marker="^", label="F1 Score")
    plt.xlabel("Confidence Threshold (tau)")
    plt.ylabel("Score")
    plt.title("Whisper Error Detection vs. Confidence Threshold (-5 dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_curve.png", dpi=200)
    plt.close()

    # 2. WER by Noise Regime
    modes = [r["noise_mode"] for r in noise_rows]
    raw_wers = [r["raw_wer"] * 100 for r in noise_rows]
    imp_wers = [r["imputed_wer"] * 100 for r in noise_rows]
    x = range(len(modes))
    width = 0.35

    plt.figure(figsize=(6.5, 4))
    plt.bar([i - width / 2 for i in x], raw_wers, width, label="Raw Faster-Whisper")
    plt.bar([i + width / 2 for i in x], imp_wers, width, label="BN Imputed (KN)")
    plt.xticks(list(x), modes)
    plt.ylabel("Word Error Rate (%)")
    plt.title("End-to-End WER Across Acoustic Degradation Regimes")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wer_by_noise.png", dpi=200)
    plt.close()

    # 3. Oracle Masking Accuracy (Phase 2 Isolated)
    labels = [
        f"{r['method'].upper()}\n({'Bi' if r['use_bidirectional'] else 'Fwd'})"
        for r in oracle_rows
    ]
    top1 = [r["top1_acc"] * 100 for r in oracle_rows]
    top5 = [r["top5_acc"] * 100 for r in oracle_rows]
    x = range(len(labels))

    plt.figure(figsize=(6.5, 4))
    plt.bar([i - width / 2 for i in x], top1, width, label="Top-1 Accuracy")
    plt.bar([i + width / 2 for i in x], top5, width, label="Top-5 Accuracy")
    plt.xticks(list(x), labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Oracle Masking Recovery Accuracy (Phase 2 Isolated)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "oracle_accuracy.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Number of utterances to evaluate (0 = all 1,680)")
    parser.add_argument("--tau", type=float, default=0.25, help="Confidence threshold for Phase 2")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading language model...")
    lm = BigramLanguageModel(vocab_size=30_000)
    lm.build_vocab(str(root / "data/unigram_freq.csv"), str(root / "data/phase1_outputs.json"))
    lm.fit(str(root / "data/sentences.txt"))
    imputer = BayesianImputer(lm)

    with open(root / "data/phase1_outputs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"Evaluating on {len(data)} utterances...")

    # Experiment 1: Threshold Calibration
    print("\n=== 1. Threshold Calibration (-5 dB) ===")
    thresh_rows = eval_threshold_sweep(data, noise_mode="noise_-5db")
    for r in thresh_rows:
        print(f"tau={r['tau']:.2f} | P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

    # Experiment 2: Smoothing Ablation (MLE vs MAP vs KN)
    print("\n=== 2. Smoothing Ablation (-5 dB, Bidirectional, aw=0.5) ===")
    smoothing_rows = []
    for method in ["mle", "map", "kn"]:
        res = evaluate_config(imputer, data, "noise_-5db", args.tau, method, True, 0.5)
        smoothing_rows.append(res)
        print(res)

    # Experiment 3: Graph Structure & Acoustic Weight Ablation (KN)
    print("\n=== 3. Structure & Acoustic Weight Ablation (-5 dB, KN) ===")
    ablation_configs = [
        (False, 0.0),  # Forward bigram only
        (True, 0.0),   # Bidirectional context only
        (True, 0.5),   # Bidirectional + gentle acoustic
        (True, 1.0),   # Bidirectional + medium acoustic
        (True, 2.0),   # Bidirectional + strong acoustic
    ]
    ablation_rows = []
    for bi, aw in ablation_configs:
        res = evaluate_config(imputer, data, "noise_-5db", args.tau, "kn", bi, aw)
        ablation_rows.append(res)
        print(res)

    best_cfg = min(ablation_rows, key=lambda r: r["imputed_wer"])
    best_bi, best_aw = best_cfg["use_bidirectional"], best_cfg["acoustic_weight"]
    print(f"\nBest config: bidirectional={best_bi}, acoustic_weight={best_aw}")

    # Experiment 4: Noise Regime Breakdown
    print("\n=== 4. Noise Regime Breakdown ===")
    noise_rows = []
    for mode in ["clean", "noise_5db", "noise_-5db", "silence"]:
        res = evaluate_config(imputer, data, mode, args.tau, "kn", best_bi, best_aw)
        noise_rows.append(res)
        print(res)

    # Experiment 5: Oracle Masking (Phase 2 Isolated)
    print("\n=== 5. Oracle Masking (Phase 2 Isolated) ===")
    oracle_configs = [
        ("mle", True),
        ("map", True),
        ("kn", False),
        ("kn", True),
    ]
    oracle_rows = []
    for method, bi in oracle_configs:
        res = eval_oracle_masking(imputer, data, method, bi)
        oracle_rows.append(res)
        print(res)

    # Save JSON + Plots
    all_results = {
        "num_utterances": len(data),
        "tau": args.tau,
        "threshold_sweep": thresh_rows,
        "smoothing_ablation": smoothing_rows,
        "structure_ablation": ablation_rows,
        "noise_breakdown": noise_rows,
        "oracle_masking": oracle_rows,
    }
    with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    save_plots(thresh_rows, noise_rows, oracle_rows, out_dir)
    print(f"\nSaved results and plots to {out_dir}/")


if __name__ == "__main__":
    main()
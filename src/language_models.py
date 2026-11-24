import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BigramLanguageModel:
    def __init__(self, vocab_size: int = 30_000):
        self.vocab_size_limit = vocab_size
        self.vocab: set[str] = set()
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.following_types: defaultdict[str, int] = defaultdict(int)
        self.preceding_types: defaultdict[str, int] = defaultdict(int)
        self.total_unique_bigrams: int = 0
        self.unigram_prior: dict[str, float] = {}

    def build_vocab(self, unigram_csv: str, phase1_json: str) -> None:
        # 1. Top-N words + raw counts from Kaggle unigram_freq.csv
        raw_counts = {}
        with open(unigram_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                word = row["word"].strip().lower()
                count = int(row["count"])
                if i < self.vocab_size_limit:
                    self.vocab.add(word)
                    raw_counts[word] = count

        # 2. Union with all TIMIT ground-truth words so no valid target is OOV
        with open(phase1_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            for utt in data:
                for w in utt["ground_truth"]:
                    self.vocab.add(w.lower())

        self.vocab.update({"<s>", "</s>"})

        # Normalized unigram prior P(w) from Kaggle frequencies (with tiny floor for unseen)
        total = sum(raw_counts.values())
        self.unigram_prior = {
            w: (raw_counts.get(w, 1) / total) for w in self.vocab
        }

    def fit(self, sentences_path: str) -> None:
        with open(sentences_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = ["<s>"] + tokenize(line) + ["</s>"]
                if len(tokens) <= 2:
                    continue
                self.unigram_counts.update(tokens)
                for w_prev, w_curr in zip(tokens[:-1], tokens[1:]):
                    self.bigram_counts[(w_prev, w_curr)] += 1

        # Precompute unique continuation counts for Kneser-Ney
        for (w_prev, w_curr) in self.bigram_counts:
            self.following_types[w_prev] += 1
            self.preceding_types[w_curr] += 1
        self.total_unique_bigrams = len(self.bigram_counts)

    def log_prob_mle(self, w: str, w_prev: str) -> float:
        """Return log P_MLE(w | w_prev), or -inf if unseen."""
        c_prev = self.unigram_counts[w_prev]
        c_bi = self.bigram_counts[(w_prev, w)]
        if c_prev == 0 or c_bi == 0:
            return float("-inf")
        return math.log(c_bi / c_prev)

    def log_prob_map(self, w: str, w_prev: str, alpha: float = 0.1) -> float:
        """Return log P_MAP(w | w_prev) with Add-alpha (Dirichlet) smoothing."""
        c_prev = self.unigram_counts[w_prev]
        c_bi = self.bigram_counts[(w_prev, w)]
        v_size = len(self.vocab)
        return math.log((c_bi + alpha) / (c_prev + alpha * v_size))

    def log_prob_kneser_ney(self, w: str, w_prev: str, d: float = 0.75) -> float:
        """Return log P_KN(w | w_prev) using Interpolated Kneser-Ney."""
        p_cont = self.preceding_types[w] / self.total_unique_bigrams
        if p_cont <= 0.0:
            p_cont = 1e-10

        c_prev = self.unigram_counts[w_prev]
        if c_prev == 0:
            return math.log(p_cont)

        c_bi = self.bigram_counts[(w_prev, w)]
        lam = (d / c_prev) * self.following_types[w_prev]
        p_kn = (max(c_bi - d, 0.0) / c_prev) + (lam * p_cont)
        return math.log(max(p_kn, 1e-10))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    lm = BigramLanguageModel(vocab_size=30_000)
    lm.build_vocab(
        str(root / "data/unigram_freq.csv"),
        str(root / "data/phase1_outputs.json"),
    )
    lm.fit(str(root / "data/sentences.txt"))

    print(f"Vocab size: {len(lm.vocab):,}")
    print(f"Unique bigrams: {lm.total_unique_bigrams:,}")

    # Sanity check: seen vs unseen pairs
    for w_prev, w in [("recognize", "speech"), ("recognize", "beach"), ("the", "xylophone")]:
        print(
            f"P({w} | {w_prev}) -> "
            f"MLE: {lm.log_prob_mle(w, w_prev):.3f}, "
            f"MAP: {lm.log_prob_map(w, w_prev):.3f}, "
            f"KN: {lm.log_prob_kneser_ney(w, w_prev):.3f}"
        )
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import jellyfish
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from language_models import BigramLanguageModel, tokenize


def compute_wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """Return (edit_distance, ref_length) using standard Levenshtein DP."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,         # deletion
                dp[i, j - 1] + 1,         # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )
    return int(dp[n, m]), n


class PhoneticViterbiHMM:
    def __init__(self, lm: BigramLanguageModel, top_k_candidates: int = 50):
        self.lm = lm
        self.top_k = top_k_candidates
        self.vocab = [w for w in lm.vocab if w not in {"<s>", "</s>"}]
        self.vocab_set = set(self.vocab)

        # Precompute phonetic codes and bucket by exact code + first character for fast lookup
        self.meta_of: dict[str, str] = {}
        self.exact_bucket: dict[str, list[str]] = defaultdict(list)
        self.prefix_bucket: dict[str, list[str]] = defaultdict(list)

        for w in self.vocab:
            code = jellyfish.metaphone(w)
            self.meta_of[w] = code
            self.exact_bucket[code].append(w)
            if code:
                self.prefix_bucket[code[0]].append(w)

        self._cand_cache: dict[str, list[tuple[str, float]]] = {}

    def _phonetic_similarity(self, cand: str, obs: str, obs_code: str) -> float:
        cand_code = self.meta_of.get(cand, jellyfish.metaphone(cand))
        phon_sim = jellyfish.jaro_winkler_similarity(cand_code, obs_code)
        char_sim = jellyfish.jaro_winkler_similarity(cand, obs)
        return 0.75 * phon_sim + 0.25 * char_sim

    def get_candidates(self, obs_word: str) -> list[tuple[str, float]]:
        """Return up to top_k phonetically closest words to obs_word with similarity scores."""
        if obs_word in self._cand_cache:
            return self._cand_cache[obs_word]

        obs_code = self.meta_of.get(obs_word, jellyfish.metaphone(obs_word))
        pool = list(self.exact_bucket.get(obs_code, []))

        # Expand to same phonetic starting sound if exact bucket is small
        if len(pool) < self.top_k and obs_code:
            seen = set(pool)
            for w in self.prefix_bucket.get(obs_code[0], []):
                if w not in seen:
                    pool.append(w)
                    seen.add(w)

        scored = [
            (w, self._phonetic_similarity(w, obs_word, obs_code))
            for w in pool
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]

        # Keep original word as a candidate (slightly lower emission if OOV so real words can win)
        if obs_word and all(w != obs_word for w, _ in top):
            orig_sim = 1.0 if obs_word in self.vocab_set else 0.75
            top.insert(0, (obs_word, orig_sim))

        self._cand_cache[obs_word] = top
        return top

    def decode_utterance(
        self,
        asr_words: list[dict],
        tau: float = 0.5,
        emission_weight: float = 4.0,
        fix_oov: bool = True,
    ) -> list[str]:
        """Run Viterbi decoding over the sentence HMM."""
        obs_seq = []
        for item in asr_words:
            toks = tokenize(item["word"])
            w = toks[0] if toks else ""
            if w:
                obs_seq.append((w, float(item["probability"])))

        if not obs_seq:
            return []

        T = len(obs_seq)

        # Build constrained hidden state space S_t for each time step t
        states: list[list[tuple[str, float]]] = []
        for w, prob in obs_seq:
            is_oov = w not in self.vocab_set
            if prob < tau or (fix_oov and is_oov):
                states.append(self.get_candidates(w))
            else:
                states.append([(w, 1.0)])

        # Viterbi DP tables: viterbi[t][j] = max log-prob ending in state j at step t
        viterbi: list[list[float]] = []
        backptr: list[list[int]] = []

        # Initialization (t = 0) from <s>
        init_scores = []
        _, p0 = obs_seq[0]
        for cand, sim in states[0]:
            trans = self.lm.log_prob_kneser_ney(cand, "<s>")
            emis = emission_weight * (1.0 + p0) * math.log(max(sim, 1e-4))
            init_scores.append(trans + emis)
        viterbi.append(init_scores)
        backptr.append([-1] * len(states[0]))

        # Recursion (t = 1 .. T-1)
        for t in range(1, T):
            _, pt = obs_seq[t]
            curr_scores = []
            curr_ptrs = []
            prev_states = states[t - 1]
            prev_v = viterbi[t - 1]

            for cand, sim in states[t]:
                emis = emission_weight * (1.0 + pt) * math.log(max(sim, 1e-4))
                best_score = float("-inf")
                best_prev = 0

                for i, (prev_cand, _) in enumerate(prev_states):
                    score = prev_v[i] + self.lm.log_prob_kneser_ney(cand, prev_cand)
                    if score > best_score:
                        best_score = score
                        best_prev = i

                curr_scores.append(best_score + emis)
                curr_ptrs.append(best_prev)

            viterbi.append(curr_scores)
            backptr.append(curr_ptrs)

        # Termination: transition to </s>
        best_final_score = float("-inf")
        best_last_idx = 0
        for j, (cand, _) in enumerate(states[T - 1]):
            score = viterbi[T - 1][j] + self.lm.log_prob_kneser_ney("</s>", cand)
            if score > best_final_score:
                best_final_score = score
                best_last_idx = j

        # Backtrack best sequence
        decoded = [""] * T
        idx = best_last_idx
        for t in range(T - 1, -1, -1):
            decoded[t] = states[t][idx][0]
            idx = backptr[t][idx]

        return decoded


def save_plot(results: dict, out_path: Path, baseline_label: str) -> None:
    regimes = ["clean", "noise_5db", "noise_-5db", "silence"]
    raw_wers = [results[r]["raw_wer"] * 100 for r in regimes]
    hmm_wers = [results[r]["viterbi_wer"] * 100 for r in regimes]

    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, raw_wers, width, label=baseline_label)
    ax.bar(x + width / 2, hmm_wers, width, label="Phonetic HMM + Viterbi")

    ax.set_ylabel("Word Error Rate (%)")
    ax.set_title(f"End-to-End WER: {baseline_label} vs. Phonetic Viterbi HMM")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="Utterances to evaluate (0 = all)")
    parser.add_argument("--tau", type=float, default=0.85, help="Confidence threshold for HMM state expansion")
    parser.add_argument("--emission_weight", type=float, default=4.0, help="Weight on phonetic emission log-prob")
    parser.add_argument("--top_k", type=int, default=50, help="Max phonetic candidates per low-confidence word")
    parser.add_argument("--input_json", type=str, default="data/phase1_outputs.json")
    parser.add_argument("--out_prefix", type=str, default="viterbi")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BigramLanguageModel...")
    lm = BigramLanguageModel(vocab_size=30_000)
    lm.build_vocab(str(root / "data/unigram_freq.csv"), str(root / "data/phase1_outputs.json"))
    lm.fit(str(root / "data/sentences.txt"))

    print("Building phonetic index...")
    hmm = PhoneticViterbiHMM(lm, top_k_candidates=args.top_k)

    with open(root / args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]

    regimes = ["clean", "noise_5db", "noise_-5db", "silence"]
    summary = {}

    for regime in regimes:
        raw_edits, hmm_edits, total_words = 0, 0, 0
        changed_words = 0

        for utt in tqdm(data, desc=f"Viterbi [{regime}]"):
            gt = [w.lower() for w in utt["ground_truth"]]
            run = utt["runs"][regime]
            raw_hyp = [tokenize(w["word"])[0] for w in run if tokenize(w["word"])]
            hmm_hyp = hmm.decode_utterance(
                run, tau=args.tau, emission_weight=args.emission_weight, fix_oov=True
            )

            r_ed, n_ref = compute_wer(gt, raw_hyp)
            h_ed, _ = compute_wer(gt, hmm_hyp)
            raw_edits += r_ed
            hmm_edits += h_ed
            total_words += n_ref
            changed_words += sum(1 for a, b in zip(raw_hyp, hmm_hyp) if a != b)

        raw_wer = raw_edits / max(total_words, 1)
        hmm_wer = hmm_edits / max(total_words, 1)
        summary[regime] = {
            "raw_wer": round(raw_wer, 4),
            "viterbi_wer": round(hmm_wer, 4),
            "delta_wer": round(hmm_wer - raw_wer, 4),
            "words_changed": changed_words,
        }
        print(
            f"{regime:10s} | Raw WER: {raw_wer*100:.2f}% | "
            f"Viterbi WER: {hmm_wer*100:.2f}% | "
            f"Delta: {(hmm_wer - raw_wer)*100:+.2f}% | Changed: {changed_words}"
        )

    json_out = out_dir / f"{args.out_prefix}_results.json"
    png_out = out_dir / f"{args.out_prefix}_wer.png"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    baseline_label = "Raw wav2vec2 (CTC)" if "wav2vec2" in args.input_json.lower() else "Raw Faster-Whisper"
    save_plot(summary, png_out, baseline_label)
    print(f"\nSaved {json_out} and {png_out}")


if __name__ == "__main__":
    main()
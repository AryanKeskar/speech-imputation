from functools import lru_cache
import json
import math
from pathlib import Path

import jellyfish

from language_models import BigramLanguageModel, tokenize


@lru_cache(maxsize=65536)
def phonetic_code(word: str) -> str:
    return jellyfish.metaphone(word)


def acoustic_log_likelihood(candidate: str, noisy_word: str, whisper_prob: float) -> float:
    if not noisy_word:
        return 0.0
    phon_sim = jellyfish.jaro_winkler_similarity(
        phonetic_code(candidate), phonetic_code(noisy_word)
    )
    char_sim = jellyfish.jaro_winkler_similarity(candidate, noisy_word)
    sim = 0.7 * phon_sim + 0.3 * char_sim
    return math.log(max(sim, 1e-4)) * whisper_prob


class BayesianImputer:
    def __init__(self, lm: BigramLanguageModel):
        self.lm = lm
        # Exclude sentence boundary tokens from candidate predictions
        self.candidates = [w for w in lm.vocab if w not in {"<s>", "</s>"}]

    def _transition_log_prob(
        self, w: str, w_prev: str, method: str, alpha: float = 0.1, d: float = 0.75
    ) -> float:
        if method == "mle":
            return self.lm.log_prob_mle(w, w_prev)
        elif method == "map":
            return self.lm.log_prob_map(w, w_prev, alpha=alpha)
        elif method == "kn":
            return self.lm.log_prob_kneser_ney(w, w_prev, d=d)
        raise ValueError(f"Unknown method: {method}")

    def _score_candidate(
        self,
        w: str,
        w_prev: str,
        w_next: str,
        noisy_word: str,
        whisper_prob: float,
        method: str,
        use_bidirectional: bool,
        acoustic_weight: float,
    ) -> float:
        score = self._transition_log_prob(w, w_prev, method=method)
        if use_bidirectional:
            score += self._transition_log_prob(w_next, w, method=method)
        if acoustic_weight > 0.0 and noisy_word:
            score += acoustic_weight * acoustic_log_likelihood(w, noisy_word, whisper_prob)
        return score

    def impute_blank(
        self,
        w_prev: str,
        w_next: str,
        noisy_word: str = "",
        whisper_prob: float = 0.0,
        method: str = "kn",
        use_bidirectional: bool = True,
        acoustic_weight: float = 0.0,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Exact inference over all candidates in V for a single blank W_t."""
        scored = [
            (
                w,
                self._score_candidate(
                    w, w_prev, w_next, noisy_word, whisper_prob,
                    method, use_bidirectional, acoustic_weight,
                ),
            )
            for w in self.candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def resolve_utterance(
        self,
        whisper_words: list[dict],
        tau: float = 0.25,
        method: str = "kn",
        use_bidirectional: bool = True,
        acoustic_weight: float = 0.0,
        margin: float = 3.0,
        gap_threshold: float = 0.35,
    ) -> tuple[list[str], list[dict]]:
        """Insert words into abnormal silence gaps and replace low-confidence words."""
        cleaned = []
        for item in whisper_words:
            toks = tokenize(item["word"])
            w = toks[0] if toks else ""
            cleaned.append((w, item["probability"], item.get("start"), item.get("end")))

        resolved = []
        imputations = []
        prev_end = None

        for i, (w, prob, start, end) in enumerate(cleaned):
            # 1. Detect deleted word via timestamp gap (primarily helps in silence regime)
            if (
                prev_end is not None
                and start is not None
                and (start - prev_end) > gap_threshold
            ):
                w_prev = resolved[-1] if resolved else "<s>"
                w_next = w if w else "</s>"
                gap_preds = self.impute_blank(
                    w_prev=w_prev,
                    w_next=w_next,
                    noisy_word="",
                    whisper_prob=0.0,
                    method=method,
                    use_bidirectional=use_bidirectional,
                    acoustic_weight=0.0,
                    top_k=5,
                )
                inserted_word = gap_preds[0][0]
                resolved.append(inserted_word)
                imputations.append({
                    "type": "gap_insertion",
                    "index": len(resolved) - 1,
                    "original_whisper": "",
                    "whisper_prob": 0.0,
                    "predicted": inserted_word,
                    "top_5": [p[0] for p in gap_preds],
                })

            # 2. Check low-confidence substitution on current word
            final_word = w
            if prob < tau:
                w_prev = resolved[-1] if resolved else "<s>"
                w_next = cleaned[i + 1][0] if i + 1 < len(cleaned) and cleaned[i + 1][0] else "</s>"
                top_preds = self.impute_blank(
                    w_prev=w_prev,
                    w_next=w_next,
                    noisy_word=w,
                    whisper_prob=prob,
                    method=method,
                    use_bidirectional=use_bidirectional,
                    acoustic_weight=acoustic_weight,
                    top_k=5,
                )
                best_word, best_score = top_preds[0]

                if w:
                    orig_score = self._score_candidate(
                        w, w_prev, w_next, w, prob,
                        method, use_bidirectional, acoustic_weight,
                    )
                else:
                    orig_score = float("-inf")

                if best_score - orig_score > margin:
                    final_word = best_word
                    imputations.append({
                        "type": "substitution",
                        "index": len(resolved),
                        "original_whisper": w,
                        "whisper_prob": prob,
                        "predicted": best_word,
                        "top_5": [p[0] for p in top_preds],
                    })

            resolved.append(final_word)
            if end is not None:
                prev_end = end

        return resolved, imputations


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    lm = BigramLanguageModel(vocab_size=30_000)
    lm.build_vocab(str(root / "data/unigram_freq.csv"), str(root / "data/phase1_outputs.json"))
    lm.fit(str(root / "data/sentences.txt"))

    imputer = BayesianImputer(lm)

    with open(root / "data/phase1_outputs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Quick qualitative check on the first 5 utterances under silence
    for utt in data[:5]:
        gt = " ".join(utt["ground_truth"])
        noisy_run = utt["runs"]["silence"]
        raw_text = " ".join(w["word"] for w in noisy_run)
        resolved, imps = imputer.resolve_utterance(
            noisy_run, tau=0.25, method="kn", use_bidirectional=True, acoustic_weight=1.0
        )
        print(f"\nID: {utt['utterance_id']} | Target: '{utt['target_word']}'")
        print(f"  Ground Truth : {gt}")
        print(f"  Raw Whisper  : {raw_text}")
        print(f"  BN Imputed   : {' '.join(resolved)}")
        print(f"  Flagged      : {imps}")
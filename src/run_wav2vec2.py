import argparse
import io
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def tokenize_gt(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def add_white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-8
    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, rms_noise, size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def mute_middle(audio: np.ndarray, sr: int = 16000, dur_sec: float = 0.4) -> np.ndarray:
    out = audio.copy().astype(np.float32)
    mid = len(out) // 2
    half_win = int((dur_sec * sr) / 2)
    start = max(0, mid - half_win)
    end = min(len(out), mid + half_win)
    out[start:end] = 0.0
    return out


def decode_audio_entry(audio_entry: dict, target_sr: int = 16000) -> np.ndarray:
    if audio_entry.get("bytes") is not None:
        audio, sr = sf.read(io.BytesIO(audio_entry["bytes"]), dtype="float32")
    elif audio_entry.get("path"):
        audio, sr = sf.read(audio_entry["path"], dtype="float32")
    else:
        raise ValueError("Audio entry has neither bytes nor path.")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


class Wav2Vec2Transcriber:
    def __init__(self):
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading facebook/wav2vec2-base-960h on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.model.eval()

        self.blank_id = self.processor.tokenizer.pad_token_id
        self.delim_id = self.processor.tokenizer.word_delimiter_token_id
        self.id_to_char = {
            idx: tok for tok, idx in self.processor.tokenizer.get_vocab().items()
        }

    def transcribe_array(self, speech: np.ndarray, sr: int = 16000) -> list[dict]:
        inputs = self.processor(
            speech.astype(np.float32), sampling_rate=sr, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
            pred_probs = probs.max(dim=-1).values.cpu().tolist()

        words = []
        curr_chars, curr_probs = [], []
        prev_id = None

        for tok_id, p in zip(pred_ids, pred_probs):
            if tok_id != prev_id:
                if tok_id == self.delim_id:
                    if curr_chars:
                        words.append({
                            "word": "".join(curr_chars).lower(),
                            "probability": float(sum(curr_probs) / len(curr_probs)),
                        })
                        curr_chars, curr_probs = [], []
                elif tok_id != self.blank_id:
                    ch = self.id_to_char.get(tok_id, "")
                    if ch and ch not in {"<pad>", "<s>", "</s>", "<unk>"}:
                        curr_chars.append(ch)
                        curr_probs.append(p)
            prev_id = tok_id

        if curr_chars:
            words.append({
                "word": "".join(curr_chars).lower(),
                "probability": float(sum(curr_probs) / len(curr_probs)),
            })

        return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kylelovesllms/timit_asr")
    parser.add_argument("--limit", type=int, default=300, help="Number of utterances (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    print(f"Loading {args.dataset} (test split) from Hugging Face...")
    ds = load_dataset(args.dataset, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    transcriber = Wav2Vec2Transcriber()
    rng = np.random.default_rng(args.seed)
    out_records = []

    for idx, row in enumerate(tqdm(ds, desc="Running wav2vec2")):
        text = row.get("text") or row.get("transcription") or row.get("sentence") or ""
        gt_words = tokenize_gt(text)
        if not gt_words:
            continue

        clean_audio = decode_audio_entry(row["audio"], target_sr=16000)
        regimes = {
            "clean": clean_audio,
            "noise_5db": add_white_noise(clean_audio, 5.0, rng),
            "noise_-5db": add_white_noise(clean_audio, -5.0, rng),
            "silence": mute_middle(clean_audio, sr=16000),
        }

        runs = {name: transcriber.transcribe_array(sig) for name, sig in regimes.items()}
        utt_id = row.get("id") or row.get("file") or f"timit_test_{idx}"
        out_records.append({
            "utterance_id": str(utt_id),
            "ground_truth": gt_words,
            "runs": runs,
        })

    out_path = root / "data/phase1_wav2vec2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    print(f"Saved {len(out_records)} utterances to {out_path}")


if __name__ == "__main__":
    main()
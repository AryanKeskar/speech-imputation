from datasets import load_dataset

ds = load_dataset("agentlans/high-quality-english-sentences", split="train")
with open("data/sentences.txt", "w") as f:
    for row in ds.select(range(200_000)):  # 200k sentences is plenty for fast bigram counts
        f.write(row["text"].strip() + "\n")
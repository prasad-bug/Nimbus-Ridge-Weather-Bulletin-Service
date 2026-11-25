import argparse, os, re
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from itertools import combinations

def words(text):
    return re.findall(r"\b[\w°.-]+\b", str(text).lower())

def pairwise_diversity(texts):
    # Average normalized edit dissimilarity over all pairs.
    # Identical responses produce 0; more different responses approach 1.
    if len(texts) < 2:
        return 0.0
    values = []
    for a, b in combinations(texts, 2):
        wa, wb = words(a), words(b)
        similarity = SequenceMatcher(None, wa, wb).ratio() if (wa or wb) else 1.0
        values.append(1.0 - similarity)
    return sum(values) / len(values)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw_log.csv")
    p.add_argument("--output", default="results")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    required = {"experiment", "varied_value", "input_id", "response_text",
                "finish_reason", "completion_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["response_text"] = df["response_text"].fillna("").astype(str)
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce")
    df["word_count"] = df["response_text"].map(lambda x: len(words(x)))
    df["truncated"] = df["finish_reason"].eq("length")

    per_input = df.groupby(
        ["experiment", "varied_value", "input_id"], dropna=False
    ).agg(
        repeats=("response_text", "size"),
        diversity=("response_text", pairwise_diversity),
        mean_word_count=("word_count", "mean"),
        mean_completion_tokens=("completion_tokens", "mean"),
        truncation_rate=("truncated", "mean")
    ).reset_index()

    summary = per_input.groupby(["experiment", "varied_value"], dropna=False).agg(
        input_groups=("input_id", "nunique"),
        total_repeats=("repeats", "sum"),
        mean_diversity=("diversity", "mean"),
        mean_word_count=("mean_word_count", "mean"),
        mean_completion_tokens=("mean_completion_tokens", "mean"),
        truncation_rate=("truncation_rate", "mean")
    ).reset_index()
    summary["truncation_rate_pct"] = summary["truncation_rate"] * 100

    os.makedirs(args.output, exist_ok=True)
    chart_dir = os.path.join(args.output, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    per_input.to_csv(os.path.join(args.output, "per_input_summary.csv"), index=False)
    summary.to_csv(os.path.join(args.output, "summary.csv"), index=False)

    for param in ["temperature", "top_p", "max_tokens", "stop", "seed"]:
        part = summary[summary["experiment"] == param].copy()
        if part.empty:
            continue
        numeric = pd.to_numeric(part["varied_value"], errors="coerce")
        part["_sort"] = numeric
        part = part.sort_values("_sort") if numeric.notna().all() else part.sort_values("varied_value")
        x = part["varied_value"].astype(str)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, col, ylabel, title in [
            (axes[0], "mean_diversity", "Mean pairwise dissimilarity (0–1)", "Diversity"),
            (axes[1], "mean_word_count", "Mean words", "Bulletin length"),
            (axes[2], "mean_completion_tokens", "Mean completion tokens", "Token usage")
        ]:
            ax.plot(x, part[col], marker="o")
            ax.set_xlabel(f"{param} setting")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{param}: diversity, length and token usage")
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, f"{param}_analysis.png"), dpi=160)
        plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved summaries and charts in {args.output}")

if __name__ == "__main__":
    main()

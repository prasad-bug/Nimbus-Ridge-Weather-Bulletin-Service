"""
Case Study 82 - Nimbus Ridge Weather Bulletin Service
API client + experiment runner.

Works with any OpenAI-compatible chat API (OpenAI, Groq, Gemini's OpenAI endpoint, etc.).

Setup:
    pip install openai
    export API_KEY="your-key"
    export MODEL="your-model-name"
    export BASE_URL="https://..."      # leave unset for OpenAI itself

Usage:
    python run_experiments.py --dry-run          # just count the calls
    python run_experiments.py --smoke            # 5 test calls
    python run_experiments.py                    # full grid (resumable)
    python run_experiments.py --workers 8        # faster, if rate limits allow

Outputs (in ./data):
    raw_log.jsonl   one JSON object per call (full parameters, text, finish reason, tokens)
    raw_log.csv     same data as CSV
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Experiment definition
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the bulletin writer for the Nimbus Ridge Weather Bulletin Service. "
    "You receive structured weather readings and write a short plain-language "
    "bulletin for villagers in hill districts. "
    "Format: one paragraph of 2 to 3 simple sentences describing the day, then a "
    "blank line, then one final line starting with 'Advisory:' that states the "
    "advisory level (write 'Advisory: none' if there is none). "
    "Use only the information in the readings. Do not add any facts, places, "
    "times or events that are not in the readings."
)

INPUTS = {
    "R1": "Max 24 C, rain 10%, wind 8 km/h, advisory: none",
    "R2": "Max 17 C, rain 85%, wind 35 km/h, advisory: landslide watch",
    "R3": "Max 29 C, rain 0%, wind 5 km/h, advisory: heat",
    "R4": "Max 12 C, rain 60%, wind 20 km/h, advisory: fog",
    "R5": "Max 21 C, rain 30%, wind 12 km/h, advisory: none",
}

# One parameter varied at a time, the rest stay at DEFAULTS.
DEFAULTS = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 200, "stop": None, "seed": None}

GRID = {
    "temperature": [0.0, 0.3, 0.7, 1.0, 1.4],
    "top_p": [0.3, 0.6, 0.9, 1.0],
    "max_tokens": [40, 80, 160],
    "stop": ["\n\n", "Advisory:"],
    "seed": [42, None],  # None = no seed; 42 = fixed seed (tests repeatability)
}

FIELDS = [
    "run_id", "timestamp", "model", "experiment", "input_id", "readings",
    "varied_value", "rep", "temperature", "top_p", "max_tokens", "stop", "seed",
    "response_text", "finish_reason", "prompt_tokens", "completion_tokens",
    "latency_s", "attempts", "error",
]


def label(value):
    if value is None:
        return "off"
    if isinstance(value, str):
        return repr(value)  # shows \n\n visibly
    return str(value)


def build_jobs(reps):
    jobs = []
    for exp, values in GRID.items():
        for value in values:
            for input_id, readings in INPUTS.items():
                for rep in range(1, reps + 1):
                    params = dict(DEFAULTS)
                    params["stop"] = [value] if exp == "stop" else None
                    if exp != "stop":
                        params[exp] = value
                    jobs.append({
                        "run_id": f"{exp}|{label(value)}|{input_id}|{rep}",
                        "experiment": exp,
                        "input_id": input_id,
                        "readings": readings,
                        "varied_value": label(value),
                        "rep": rep,
                        "params": params,
                    })
    return jobs


# ----------------------------------------------------------------------------
# API call with retry, timeout and full logging
# ----------------------------------------------------------------------------
def call_api(client, model, job, timeout, max_attempts=6, delay=0.0):
    import openai

    p = job["params"]
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Readings: {job['readings']}\nWrite the bulletin."},
        ],
        "temperature": p["temperature"],
        "top_p": p["top_p"],
        "max_tokens": p["max_tokens"],
        "timeout": timeout,
    }
    if p["stop"]:
        kwargs["stop"] = p["stop"]
    if p["seed"] is not None:
        kwargs["seed"] = p["seed"]

    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    row = {
        "run_id": job["run_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "experiment": job["experiment"],
        "input_id": job["input_id"],
        "readings": job["readings"],
        "varied_value": job["varied_value"],
        "rep": job["rep"],
        "temperature": p["temperature"],
        "top_p": p["top_p"],
        "max_tokens": p["max_tokens"],
        "stop": p["stop"],
        "seed": p["seed"],
        "response_text": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_s": None,
        "attempts": 0,
        "error": None,
    }

    for attempt in range(1, max_attempts + 1):
        row["attempts"] = attempt
        if delay:
            time.sleep(delay)  # spacing between calls to stay under rate limits
        start = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            row["latency_s"] = round(time.time() - start, 3)
            choice = resp.choices[0]
            row["response_text"] = choice.message.content
            row["finish_reason"] = choice.finish_reason
            if resp.usage:
                row["prompt_tokens"] = resp.usage.prompt_tokens
                row["completion_tokens"] = resp.usage.completion_tokens
            row["error"] = None
            return row
        except retryable as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if attempt < max_attempts:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))  # backoff: 5s, 10s, 20s, 40s, 60s
        except Exception as e:  # bad request, auth error, etc. -> do not retry
            row["latency_s"] = round(time.time() - start, 3)
            row["error"] = f"{type(e).__name__}: {e}"
            return row
    return row


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    done.add(r["run_id"])
    return done


def jsonl_to_csv(jsonl_path, csv_path):
    # keep only the latest record per run_id (so retries of failed runs replace errors)
    latest = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[r["run_id"]] = r
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in latest.values():
            r = dict(r)
            r["stop"] = json.dumps(r["stop"]) if r.get("stop") else ""
            w.writerow({k: r.get(k) for k in FIELDS})
    return len(latest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5, help="repetitions per setting (default 5)")
    ap.add_argument("--workers", type=int, default=5, help="parallel threads (default 5)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds each worker waits before every call")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-call timeout in seconds")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--dry-run", action="store_true", help="only print the number of calls")
    ap.add_argument("--smoke", action="store_true", help="run just 5 calls to test setup")
    args = ap.parse_args()

    jobs = build_jobs(args.reps)
    if args.dry_run:
        counts = {}
        for j in jobs:
            counts[j["experiment"]] = counts.get(j["experiment"], 0) + 1
        print(f"Total calls: {len(jobs)}")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        return
    if args.smoke:
        jobs = jobs[:5]

    from openai import OpenAI

    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("MODEL")
    base_url = os.environ.get("BASE_URL") or None
    if not api_key or not model:
        sys.exit("Set API_KEY and MODEL (and BASE_URL if not using OpenAI) first.")
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    os.makedirs(args.out, exist_ok=True)
    jsonl_path = os.path.join(args.out, "raw_log.jsonl")
    csv_path = os.path.join(args.out, "raw_log.csv")

    done = load_done(jsonl_path)
    todo = [j for j in jobs if j["run_id"] not in done]
    print(f"{len(jobs)} planned, {len(done)} already done, {len(todo)} to run "
          f"with {args.workers} workers, model={model}")

    lock = threading.Lock()
    completed = errors = 0
    with open(jsonl_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(call_api, client, model, j, args.timeout, 6, args.delay) for j in todo]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                if row["error"]:
                    errors += 1
                    print(f"[{completed}/{len(todo)}] ERROR {row['run_id']}: {row['error'][:120]}")
                elif completed % 25 == 0 or completed == len(todo):
                    print(f"[{completed}/{len(todo)}] done, {errors} errors")

    n = jsonl_to_csv(jsonl_path, csv_path)
    print(f"Wrote {n} rows to {csv_path}")
    if errors:
        print(f"{errors} calls failed. Re-run the same command to retry only the failed ones.")


if __name__ == "__main__":
    main()

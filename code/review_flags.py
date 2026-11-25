"""Automated flag pass over every response in raw_log.csv.
Usage: python3 -I code/review_flags.py data/raw_log.csv results
Flags are keyword/regex heuristics (approximate); the manual review CSV is the ground truth for the 20 reviewed rows."""
import csv, re, sys, os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(raw, encoding="utf-8")))
REQ = {"R2": "landslide", "R3": "heat", "R4": "fog"}
PAT = {
 "invented_sky":   r"clear sk|sunny|overcast|mostly cloudy|partly cloudy|predominantly cloudy|\bcloudy\b|\bsun\b|bright",
 "humidity_mislabel": r"humid",
 "invented_wind_dir": r"from the (north|south|east|west)|northwest|southwest",
 "invented_time":  r"morning|afternoon|evening|sunset|noontime|tonight",
 "future_promise": r"tomorrow|stay tuned|tuning in",
 "invented_storm": r"storm|thunder|hail|snow",
 "non_english":    r"[؀-ۿ]",
}
flag_names = list(PAT) + ["extra_number", "missing_reading", "truncated", "missing_required_advisory"]
flagged = []
for i, x in enumerate(rows):
    t = x["response_text"]
    f = {k: int(bool(re.search(p, t, re.I))) for k, p in PAT.items()}
    f["extra_number"] = int(bool(set(re.findall(r"\d+", t)) - set(re.findall(r"\d+", x["readings"]))))
    need_nums = set(re.findall(r"\d+", x["readings"])) - {"0"}  # a 0% rain chance may be written as "no rain"
    f["missing_reading"] = int(x["finish_reason"] != "length" and not need_nums <= set(re.findall(r"\d+", t)))
    f["truncated"] = int(x["finish_reason"] == "length")
    f["missing_required_advisory"] = int(x["input_id"] in REQ and REQ[x["input_id"]] not in t.lower())
    f["any_flag"] = int(any(f.values()))
    flagged.append({"row": i, "experiment": x["experiment"], "varied_value": x["varied_value"],
                    "input_id": x["input_id"], "rep": x["rep"], "finish_reason": x["finish_reason"],
                    **f, "response_text": t})
cols = list(flagged[0].keys())
with open(os.path.join(out, "flagged_all_responses.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(flagged)

g = collections.OrderedDict()
for f in flagged:
    g.setdefault((f["experiment"], f["varied_value"]), []).append(f)
summ = []
for (e, v), L in g.items():
    d = {"experiment": e, "varied_value": v, "n": len(L)}
    for k in flag_names: d[k + "_pct"] = round(100 * sum(z[k] for z in L) / len(L), 1)
    summ.append(d)
with open(os.path.join(out, "flag_rates_by_setting.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

# charts
def chart(exp, fname, keys, title):
    S = [s for s in summ if s["experiment"] == exp]
    labels = [s["varied_value"] for s in S]
    import numpy as np
    xs = np.arange(len(S)); wd = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, k in enumerate(keys):
        ax.bar(xs + j * wd, [s[k + "_pct"] for s in S], wd, label=k.replace("_", " "))
    ax.set_xticks(xs + wd * (len(keys) - 1) / 2); ax.set_xticklabels(labels)
    ax.set_ylabel("% of responses"); ax.set_xlabel(exp); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "charts", fname), dpi=140); plt.close(fig)
chart("max_tokens", "truncation_by_max_tokens.png", ["truncated", "missing_required_advisory"], "Truncation and missing advisory by max_tokens")
chart("temperature", "flags_by_temperature.png", ["invented_sky", "non_english", "extra_number", "humidity_mislabel"], "Invented / garbled content by temperature")
chart("stop", "advisory_loss_by_stop.png", ["missing_required_advisory", "truncated"], "Missing required advisory by stop sequence")
print("done", len(flagged))

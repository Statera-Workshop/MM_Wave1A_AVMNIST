"""Wave 1-A analysis. Reads runs/A/*/curve.jsonl and prints a Markdown table.

Per run : final test acc (last epoch), E90 / E99 (first epoch reaching 90% / 99% of final), peak,
          peak epoch and drop (peak - final), AUC (mean of
          test acc over epochs 1..E), unconverged flag (last 3 evals rising and +0.01 or more),
          final linear-probe accuracy per modality and imbalance gap (unimodal model's own probe
          minus the fusion model's probe, same budget, seed-mean).
Per group (model, opt, epochs): mean over seeds and seed range (max - min).
Writes summary.json next to the runs.
"""
import argparse, glob, json, os
from collections import defaultdict


def load(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "curve.jsonl")) if l.strip()]
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    fin = os.path.join(run_dir, "final.json")
    status = json.load(open(fin)).get("status") if os.path.exists(fin) else "running"
    return meta, rows, status


def metrics(rows, epochs, key="test_acc"):
    cur = [r for r in rows if r["epoch"] >= 1]
    if not cur:
        return None
    vals = [r[key] for r in cur]
    final = vals[-1]
    e90 = next((r["epoch"] for r in cur if r[key] >= 0.9 * final), None)
    auc = sum(vals) / max(epochs, 1)
    tail = vals[-3:]
    unconv = len(tail) == 3 and tail[0] < tail[1] < tail[2] and tail[2] - tail[0] >= 0.01
    peak = max(vals)
    peak_ep = cur[vals.index(peak)]["epoch"]
    e99 = next((r["epoch"] for r in cur if r[key] >= 0.99 * final), None)
    return {"final": final, "e90": e90, "e99": e99, "auc": auc, "unconverged": unconv,
            "peak": peak, "peak_epoch": peak_ep, "drop": peak - final, "last_epoch": cur[-1]["epoch"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.expanduser("~/MM_Wave1/runs/A"))
    a = p.parse_args()
    per_run, groups = {}, defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.runs, "*", "curve.jsonl"))):
        rd = os.path.dirname(d)
        meta, rows, status = load(rd)
        ar = meta["args"]
        m = metrics(rows, ar["epochs"])
        if m is None:
            continue
        last = rows[-1]
        m.update(status=status, model=ar["model"], opt=ar["opt"], epochs=ar["epochs"], seed=ar["seed"],
                 probe_img=last.get("probe_img"), probe_aud=last.get("probe_aud"),
                 featstd_img=last.get("featstd_img"), featstd_aud=last.get("featstd_aud"))
        per_run[meta["name"]] = m
        groups[(ar["model"], ar["opt"], ar["epochs"])].append(m)

    # imbalance gap reference = the unimodal model's own linear probe (same probe as the fusion models),
    # so the gap does not mix probe-vs-trained-head differences into the comparison
    ref = {}
    for (model, opt, ep), ms in groups.items():
        if model in ("img", "aud") and opt == "adam":
            ref[(model, ep)] = sum(x[f"probe_{model}"] for x in ms) / len(ms)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def f(x, nd=4):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    summary = []
    print("| model | opt | epochs | seeds | final (mean) | seed range | peak | peak ep | drop | E99 | AUC "
          "| probe img | probe aud | gap img | gap aud | flags |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {k: i for i, k in enumerate(["img", "aud", "late", "late_lin", "tfn", "lmf", "gblend", "ogmge", "ogm"])}
    for key in sorted(groups, key=lambda k: (k[2], order.get(k[0], 99), k[1])):
        model, opt, ep = key
        ms = groups[key]
        finals = [x["final"] for x in ms]
        pi, pa = mean([x["probe_img"] for x in ms]), mean([x["probe_aud"] for x in ms])
        gi = ref.get(("img", ep)) - pi if (pi is not None and ("img", ep) in ref) else None
        ga = ref.get(("aud", ep)) - pa if (pa is not None and ("aud", ep) in ref) else None
        flags = []
        if any(x["unconverged"] for x in ms):
            flags.append("unconverged")
        if any(x["status"] != "ok" for x in ms):
            flags.append(",".join(sorted({x["status"] for x in ms if x["status"] != "ok"})))
        if any((x["probe_img"] is not None and x["probe_img"] <= 0.15) or
               (x["probe_aud"] is not None and x["probe_aud"] <= 0.15) for x in ms):
            flags.append("modality-collapse")
        row = {"model": model, "opt": opt, "epochs": ep, "seeds": sorted(x["seed"] for x in ms),
               "final_mean": mean(finals), "seed_range": max(finals) - min(finals),
               "e90_mean": mean([x["e90"] for x in ms]), "e99_mean": mean([x["e99"] for x in ms]),
               "auc_mean": mean([x["auc"] for x in ms]), "peak_mean": mean([x["peak"] for x in ms]),
               "peak_epoch_mean": mean([x["peak_epoch"] for x in ms]), "drop_mean": mean([x["drop"] for x in ms]),
               "probe_img": pi, "probe_aud": pa, "gap_img": gi, "gap_aud": ga, "flags": flags}
        summary.append(row)
        print(f"| {model} | {opt} | {ep} | {len(ms)} | {f(row['final_mean'])} | {f(row['seed_range'])} | "
              f"{f(row['peak_mean'])} | {f(row['peak_epoch_mean'], 1)} | {f(row['drop_mean'])} | "
              f"{f(row['e99_mean'], 1)} | {f(row['auc_mean'])} | {f(pi)} | {f(pa)} | {f(gi)} | {f(ga)} | "
              f"{' '.join(flags) or '-'} |")
    json.dump({"groups": summary, "runs": per_run}, open(os.path.join(a.runs, "summary.json"), "w"), indent=1)
    print(f"\n{len(per_run)} runs, summary.json written to {a.runs}")


if __name__ == "__main__":
    main()

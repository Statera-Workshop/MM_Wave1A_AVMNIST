"""Build the Wave 1-A queue from smoke timings.

Reads runs/A_smoke/<model>_<opt>_e1_s777/curve.jsonl (epoch_sec + eval_sec of epoch 1) and writes
'name | est_min | command' lines. Order: 30-epoch budget for every model and seed first, then
100 epochs, so a stop at the deadline still leaves complete sets.
"""
import argparse, json, os

MODELS = ["img", "aud", "late", "late_lin", "tfn", "lmf", "gblend", "ogmge"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", default=os.path.expanduser("~/MM_Wave1/runs/A_smoke"))
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/queue/A_main.txt"))
    p.add_argument("--budgets", default="30,100")
    p.add_argument("--seeds", default="777,778")
    p.add_argument("--margin", type=float, default=1.2)
    a = p.parse_args()

    per_epoch = {}
    for m in MODELS:
        f = os.path.join(a.smoke, f"{m}_adam_e1_s777", "curve.jsonl")
        rows = [json.loads(l) for l in open(f) if l.strip()]
        r = [x for x in rows if x["epoch"] == 1][0]
        per_epoch[m] = r["epoch_sec"] + r["eval_sec"]
        print(f"{m:8s} train {r['epoch_sec']:.1f}s + eval {r['eval_sec']:.1f}s per epoch, "
              f"mem {r.get('max_mem_mb', 0):.0f} MB")

    jobs = []
    for ep in [int(x) for x in a.budgets.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            for m in MODELS:
                jobs.append((m, "adam", ep, seed))
            jobs.append(("ogmge", "sgd_official", ep, seed))
    lines, total = [], 0.0
    for m, opt, ep, seed in jobs:
        name = f"A_{m}_{opt}_e{ep}_s{seed}"
        est = max(1, round(per_epoch[m] * ep * a.margin / 60))
        total += est
        cmd = (f"cd ~/MM_Wave1/code && python mm_a.py --model {m} --opt {opt} --epochs {ep} --seed {seed}")
        lines.append(f"{name} | {est} | {cmd}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs, estimated {total / 60:.1f} h -> {a.out}")


if __name__ == "__main__":
    main()

"""Re-evaluate saved Wave 1-A models with recalibrated BatchNorm statistics.

Same idea as recal_bn_b.py: evaluate last.pt as saved ("before", must equal final.json), recompute every
BatchNorm running mean/var as a cumulative average over --n-batches training batches (batch 64, both
modalities, train mode, no gradient), evaluate again ("after"). Writes <run>/recal.json and a table.
"""
import argparse, glob, json, os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_a as M


@torch.no_grad()
def recalibrate(model, tr, n_batches, bs, seed):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(tr["y"]), generator=g)[: n_batches * bs].to(tr["y"].device)
    model.train()
    for s in range(0, idx.numel(), bs):
        img, aud, _ = M.get_batch(tr, idx[s:s + bs])
        model(img, aud)
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.expanduser("~/MM_Wave1/runs/A/*"))
    p.add_argument("--data", default=os.path.expanduser("~/MM_Wave1/data/avmnist"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-batches", type=int, default=200)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    dev = torch.device(a.device)
    tr = va = te = None
    print("| run | final.json | before | after | change | probe aud before -> after |")
    print("|---|---|---|---|---|---|")
    n = 0
    for rd in sorted(glob.glob(a.runs)):
        if not os.path.exists(os.path.join(rd, "last.pt")):
            continue
        out = os.path.join(rd, "recal.json")
        if os.path.exists(out) and not a.force:
            r = json.load(open(out))
        else:
            if tr is None:
                tr, va, te, _ = M.load_avmnist(a.data, dev)
            meta = json.load(open(os.path.join(rd, "meta.json")))
            ar = meta["args"]
            model = M.AVNet(ar["model"]).to(dev)
            model.load_state_dict(torch.load(os.path.join(rd, "last.pt"), map_location=dev, weights_only=True))
            model.eval()
            fin = json.load(open(os.path.join(rd, "final.json"))).get("test_acc")
            before = M.evaluate(model, tr, va, te, ar["probe_n"])
            recalibrate(model, tr, a.n_batches, 64, ar["seed"])
            after = M.evaluate(model, tr, va, te, ar["probe_n"])
            r = {"name": meta["name"], "final_json": fin, "before": before, "after": after, "n_batches": a.n_batches}
            json.dump(r, open(out, "w"), indent=1)
        b, af = r["before"]["test_acc"], r["after"]["test_acc"]
        pa = (f"{r['before']['probe_aud']:.4f} -> {r['after']['probe_aud']:.4f}"
              if "probe_aud" in r["before"] else "-")
        print(f"| {r['name']} | {r['final_json']:.4f} | {b:.4f} | {af:.4f} | {af - b:+.4f} | {pa} |", flush=True)
        n += 1
    print(f"\n{n} runs re-evaluated")


if __name__ == "__main__":
    main()

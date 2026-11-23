"""Wave 1-A gates (run before any full training).

1. eval path : one-hot logits -> accuracy 1.000; one-hot features -> linear probe 1.000.
2. params    : every parameter of every model is in model.parameters() and changes after
               one optimizer step (catches MultiBench's unregistered LowRankTensorFusion factors).
3. overfit   : 256 training pairs, full batch, eval-mode accuracy 1.000 within --steps.
Prints one line per check and 'GATE A: PASS' / 'GATE A: FAIL' at the end.
"""
import argparse, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_a as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/MM_Wave1/data/avmnist"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--models", default=",".join(M.KINDS))
    p.add_argument("--expect-train", type=int, default=60000)
    p.add_argument("--expect-test", type=int, default=10000)
    a = p.parse_args()
    dev = torch.device(a.device)
    ok = True
    tr, va, te, ds = M.load_avmnist(a.data, dev, a.expect_train, a.expect_test)
    print(f"data train {ds['n_train']} val {ds['n_val']} test {ds['n_test']} img mean {ds['img_mean']:.3f} std {ds['img_std']:.3f}")

    # 1. eval path
    y = te["y"]
    acc = M.accuracy(F.one_hot(y, 10).float(), y)
    sub = M.subset(tr, 10000)
    pr = M.linear_probe(F.one_hot(sub["y"], 10).float(), sub["y"], F.one_hot(y, 10).float(), y)
    good = acc == 1.0 and pr == 1.0
    ok &= good
    print(f"[1] eval path: accuracy {acc:.3f} probe {pr:.3f} -> {'OK' if good else 'FAIL'}")

    # 2. parameter registration and update
    for kind in a.models.split(","):
        M.seed_all(777)
        m = M.AVNet(kind).to(dev)
        opt, _ = M.make_opt(m, "adam")
        before = {n: q.detach().clone() for n, q in m.named_parameters()}
        st = {"alpha": 0.1, "mod_start": 0, "mod_end": 50, "epoch": 0,
              "gb_w": {"fused": 1 / 3, "img": 1 / 3, "aud": 1 / 3}}
        img, aud, yy = M.get_batch(tr, torch.arange(64, device=dev))
        M.train_step(m, opt, img, aud, yy, st)
        stale = [n for n, q in m.named_parameters() if torch.equal(before[n], q.detach())]
        n_par = sum(q.numel() for q in m.parameters())
        extra = ""
        if kind == "lmf":
            extra = f" fusion params {sum(q.numel() for q in m.fusion.parameters())}"
        good = not stale
        ok &= good
        print(f"[2] {kind:8s} params {n_par:>9,}{extra} unchanged {stale if stale else 0} -> {'OK' if good else 'FAIL'}")

    # 3. overfit
    idx = torch.arange(a.n, device=dev)
    img, aud, yy = M.get_batch(tr, idx)
    for kind in a.models.split(","):
        M.seed_all(777)
        m = M.AVNet(kind).to(dev)
        opt, _ = M.make_opt(m, "adam")
        st = {"alpha": 0.1, "mod_start": 0, "mod_end": 50, "epoch": 0,
              "gb_w": {"fused": 1 / 3, "img": 1 / 3, "aud": 1 / 3}}
        t0 = time.time()
        hit = None
        for s in range(1, a.steps + 1):
            r = M.train_step(m, opt, img, aud, yy, st)
            if s % 50 == 0:
                m.eval()
                with torch.no_grad():
                    acc = M.accuracy(m(img, aud)[0], yy)
                if acc == 1.0:
                    hit = s
                    break
        good = hit is not None
        ok &= good
        print(f"[3] {kind:8s} overfit {a.n}: acc 1.000 at step {hit} loss {r['loss']:.4f} "
              f"{(time.time() - t0):.1f}s -> {'OK' if good else 'FAIL (acc ' + format(acc, '.3f') + ')'}")
    print("GATE A:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

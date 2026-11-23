"""PAI-SG Multimodal Wave 1-A: AV-MNIST fusion convergence runs.

Models : img, aud (unimodal references), late (MLP head), late_lin (linear head,
         OGM-GE's 'Normal' control), tfn, lmf, gblend, ogmge, ogm (official 'OGM' mode: no GE noise).
Sources: LeNet / MLP / Concat / TensorFusion / LowRankTensorFusion follow MultiBench (MIT).
         OGM-GE follows GeWu-Lab/OGM-GE_CVPR2022 main.py (MIT).
         Gradient Blending is implemented from the paper (Wang, Tran, Feiszli, CVPR 2020).

[Unified]   image standardised with the mean/std of the 55,000 training pairs
            (MultiBench divides by 255, but these images are PCA-processed floats).
[Unified]   unimodal references use the same encoders as the fusion models (MultiBench's
            unimodal image example uses 3 channels; here 6 everywhere).
[Unified]   Adam 1e-3 constant, batch 64 for every model; --opt sgd_official reproduces
            OGM-GE's SGD(1e-3, momentum 0.9, wd 1e-4) + StepLR(70, 0.1).
[Framework] LeNet returns torch.flatten(x, 1) instead of squeeze() (safe for batch 1).
[Framework] LowRankTensorFusion registers its factors as nn.Parameter. In MultiBench they are
            kept in a Python list and moved with .to(), so on GPU none of the fusion tensors
            reach the optimizer; the gate checks that every parameter is trained here.
[Unified]   Gradient Blending estimates its weights on the 5,000-pair val split (held out from
            the training file, as the paper holds out part of the training set).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_VAL = 5000
D_IMG, D_AUD, N_CLS = 48, 192, 10
KINDS = ["img", "aud", "late", "late_lin", "tfn", "lmf", "gblend", "ogmge", "ogm"]


# ----------------------------------------------------------------------------- data
def load_avmnist(d, device, expect_train=60000, expect_test=10000):
    im_tr = np.load(f"{d}/image/train_data.npy").astype(np.float32).reshape(-1, 1, 28, 28)
    au_tr = np.load(f"{d}/audio/train_data.npy").reshape(-1, 1, 112, 112)
    y_tr = np.load(f"{d}/train_labels.npy").astype(np.int64)
    im_te = np.load(f"{d}/image/test_data.npy").astype(np.float32).reshape(-1, 1, 28, 28)
    au_te = np.load(f"{d}/audio/test_data.npy").reshape(-1, 1, 112, 112)
    y_te = np.load(f"{d}/test_labels.npy").astype(np.int64)
    assert len(im_tr) == len(au_tr) == len(y_tr) == expect_train, "train count mismatch"
    assert len(im_te) == len(au_te) == len(y_te) == expect_test, "test count mismatch"
    assert au_tr.dtype == np.uint8, "audio expected uint8"
    n_tr = expect_train - N_VAL
    mu, sd = float(im_tr[:n_tr].mean()), float(im_tr[:n_tr].std())

    def pack(im, au, y):
        return {"img": torch.from_numpy((im - mu) / sd).to(device),
                "aud": torch.from_numpy(np.ascontiguousarray(au)).to(device),
                "y": torch.from_numpy(y).to(device)}

    tr = pack(im_tr[:n_tr], au_tr[:n_tr], y_tr[:n_tr])
    va = pack(im_tr[n_tr:], au_tr[n_tr:], y_tr[n_tr:])
    te = pack(im_te, au_te, y_te)
    return tr, va, te, {"img_mean": mu, "img_std": sd, "n_train": n_tr, "n_val": N_VAL, "n_test": expect_test}


def get_batch(ds, idx):
    return ds["img"][idx], ds["aud"][idx].float().div_(255.0), ds["y"][idx]


# ----------------------------------------------------------------------------- models
class LeNet(nn.Module):
    """MultiBench LeNet (MIT). Image: LeNet(1,6,3) -> 48. Audio: LeNet(1,6,5) -> 192."""

    def __init__(self, cin, ch, extra):
        super().__init__()
        convs = [nn.Conv2d(cin, ch, 5, padding=2, bias=False)]
        bns = [nn.BatchNorm2d(ch)]
        for i in range(extra):
            convs.append(nn.Conv2d(ch * 2 ** i, ch * 2 ** (i + 1), 3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(ch * 2 ** (i + 1)))
        self.convs, self.bns = nn.ModuleList(convs), nn.ModuleList(bns)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        for c, b in zip(self.convs, self.bns):
            x = F.max_pool2d(F.relu(b(c(x))), 2)
        return torch.flatten(x, 1)


class MLP(nn.Module):
    """MultiBench MLP (MIT): fc -> relu -> fc2."""

    def __init__(self, i, h, o):
        super().__init__()
        self.fc, self.fc2 = nn.Linear(i, h), nn.Linear(h, o)

    def forward(self, x):
        return self.fc2(F.relu(self.fc(x)))


def tensor_fusion(a, b):
    """MultiBench TensorFusion (MIT): outer product of [1, a] and [1, b]."""
    one = torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)
    a1, b1 = torch.cat([one, a], 1), torch.cat([one, b], 1)
    return torch.einsum("ni,nj->nij", a1, b1).reshape(a.shape[0], -1)


class LowRankTensorFusion(nn.Module):
    """MultiBench LowRankTensorFusion (MIT) with factors registered as parameters [Framework]."""

    def __init__(self, input_dims, output_dim, rank):
        super().__init__()
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, d + 1, output_dim)) for d in input_dims])
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, output_dim))
        for f in self.factors:
            nn.init.xavier_normal_(f)
        nn.init.xavier_normal_(self.fusion_weights)
        self.output_dim = output_dim

    def forward(self, mods):
        n = mods[0].shape[0]
        fused = 1
        for m, f in zip(mods, self.factors):
            one = torch.ones(n, 1, dtype=m.dtype, device=m.device)
            fused = fused * torch.matmul(torch.cat([one, torch.flatten(m, 1)], 1), f)  # (rank, n, out)
        out = torch.matmul(self.fusion_weights, fused.permute(1, 0, 2)).squeeze(1) + self.fusion_bias
        return out.view(-1, self.output_dim)


class AVNet(nn.Module):
    def __init__(self, kind):
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        self.enc_img = LeNet(1, 6, 3) if kind != "aud" else None
        self.enc_aud = LeNet(1, 6, 5) if kind != "img" else None
        if kind == "img":
            self.head = MLP(D_IMG, 100, N_CLS)
        elif kind == "aud":
            self.head = MLP(D_AUD, 100, N_CLS)
        elif kind in ("late", "gblend"):
            self.head = MLP(D_IMG + D_AUD, 100, N_CLS)
        elif kind == "tfn":
            self.head = MLP((D_IMG + 1) * (D_AUD + 1), 100, N_CLS)
        elif kind == "lmf":  # MultiBench avmnist_low_rank_tensor: out 120, rank 40, head MLP(120,100,10)
            self.fusion = LowRankTensorFusion([D_IMG, D_AUD], 120, 40)
            self.head = MLP(120, 100, N_CLS)
        elif kind in ("late_lin", "ogmge", "ogm"):  # OGM-GE official concat: one linear layer on [img, aud]
            self.head = nn.Linear(D_IMG + D_AUD, N_CLS)
        if kind == "gblend":
            self.uni_img, self.uni_aud = nn.Linear(D_IMG, N_CLS), nn.Linear(D_AUD, N_CLS)

    def forward(self, img, aud):
        fi = self.enc_img(img) if self.enc_img is not None else None
        fa = self.enc_aud(aud) if self.enc_aud is not None else None
        k = self.kind
        if k == "img":
            out = self.head(fi)
        elif k == "aud":
            out = self.head(fa)
        elif k == "tfn":
            out = self.head(tensor_fusion(fi, fa))
        elif k == "lmf":
            out = self.head(self.fusion([fi, fa]))
        else:
            out = self.head(torch.cat([fi, fa], 1))
        return out, fi, fa


# ----------------------------------------------------------------------------- training
def make_opt(model, opt_name):
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=1e-3), None
    if opt_name == "sgd_official":
        o = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        return o, torch.optim.lr_scheduler.StepLR(o, 70, 0.1)
    raise ValueError(opt_name)


def ogm_coeffs(model, fi, fa, y, alpha):
    """OGM-GE official: per-modality scores from the split linear head, coefficient
    1 - tanh(alpha * relu(ratio)) for the dominant modality (code form, not paper's rho-1)."""
    with torch.no_grad():
        W, b = model.head.weight, model.head.bias
        out_i = fi @ W[:, :D_IMG].t() + b / 2
        out_a = fa @ W[:, D_IMG:].t() + b / 2
        s_i = F.softmax(out_i, 1).gather(1, y[:, None]).sum()
        s_a = F.softmax(out_a, 1).gather(1, y[:, None]).sum()
        r_i = (s_i / s_a).item()
        r_a = 1.0 / r_i
        if r_i > 1:
            c_i, c_a = 1 - math.tanh(alpha * max(r_i, 0.0)), 1.0
        else:
            c_i, c_a = 1.0, 1 - math.tanh(alpha * max(r_a, 0.0))
    return r_i, c_i, c_a


def train_step(model, opt, img, aud, y, st):
    model.train()
    opt.zero_grad(set_to_none=True)
    out, fi, fa = model(img, aud)
    loss = F.cross_entropy(out, y)
    rec = {"loss": loss.item()}
    if model.kind == "gblend":
        w = st["gb_w"]
        li = F.cross_entropy(model.uni_img(fi), y)
        la = F.cross_entropy(model.uni_aud(fa), y)
        (w["fused"] * loss + w["img"] * li + w["aud"] * la).backward()
    else:
        loss.backward()
    if model.kind in ("ogmge", "ogm"):
        r_i, c_i, c_a = ogm_coeffs(model, fi.detach(), fa.detach(), y, st["alpha"])
        rec.update(ratio_img=r_i, coeff_img=c_i, coeff_aud=c_a)
        if st["mod_start"] <= st["epoch"] <= st["mod_end"]:
            for enc, c in ((model.enc_img, c_i), (model.enc_aud, c_a)):
                for p in enc.parameters():
                    if p.grad is not None and p.grad.dim() == 4:  # conv weights, as in the official code
                        if model.kind == "ogmge":  # official 'OGM_GE': scale + Gaussian noise (both encoders)
                            p.grad = p.grad * c + torch.zeros_like(p.grad).normal_(0, p.grad.std().item() + 1e-8)
                        else:  # official 'OGM': scale only
                            p.grad = p.grad * c
    opt.step()
    return rec


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, ds, bs=1000):
    model.eval()
    outs, fis, fas, extra = [], [], [], {"img": [], "aud": []}
    n = len(ds["y"])
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n), device=ds["y"].device)
        img, aud, _ = get_batch(ds, idx)
        out, fi, fa = model(img, aud)
        outs.append(out)
        fis.append(fi)
        fas.append(fa)
        if model.kind == "gblend":
            extra["img"].append(model.uni_img(fi))
            extra["aud"].append(model.uni_aud(fa))
    cat = lambda xs: torch.cat(xs) if xs and xs[0] is not None else None
    return cat(outs), cat(fis), cat(fas), {k: cat(v) for k, v in extra.items()}


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def linear_probe(ftr, ytr, fte, yte, steps=300, lr=0.05):
    """Frozen-feature linear probe: standardise with train stats, full-batch Adam from zero init."""
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    a, b = (ftr - mu) / sd, (fte - mu) / sd
    W = torch.zeros(a.shape[1], N_CLS, device=a.device, requires_grad=True)
    c = torch.zeros(N_CLS, device=a.device, requires_grad=True)
    opt = torch.optim.Adam([W, c], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(a @ W + c, ytr).backward()
            opt.step()
    with torch.no_grad():
        return accuracy(b @ W + c, yte)


@torch.no_grad()
def gb_losses(model, ds_tr_sub, ds_va):
    r = {}
    for name, ds in (("t", ds_tr_sub), ("v", ds_va)):
        out, _, _, ex = predict(model, ds)
        y = ds["y"]
        r[name] = {"fused": F.cross_entropy(out, y).item(),
                   "img": F.cross_entropy(ex["img"], y).item(),
                   "aud": F.cross_entropy(ex["aud"], y).item()}
    return r


def gb_weights(prev, cur):
    """Paper Eq.: w_k proportional to dG_k / dO_k^2 between two checkpoints."""
    raw = {}
    for k in ("fused", "img", "aud"):
        dG = prev["v"][k] - cur["v"][k]
        dO = (cur["v"][k] - cur["t"][k]) - (prev["v"][k] - prev["t"][k])
        raw[k] = max(dG, 0.0) / max(dO * dO, 1e-6)
    z = sum(raw.values())
    if z <= 0 or not math.isfinite(z):
        return {k: 1 / 3 for k in raw}, raw
    return {k: v / z for k, v in raw.items()}, raw


def subset(ds, n):
    return {k: v[:n] for k, v in ds.items()}


def evaluate(model, tr, va, te, probe_n):
    out_te, fi_te, fa_te, _ = predict(model, te)
    out_va, _, _, _ = predict(model, va)
    rec = {"test_acc": accuracy(out_te, te["y"]), "val_acc": accuracy(out_va, va["y"]),
           "val_loss": F.cross_entropy(out_va, va["y"]).item()}
    sub = subset(tr, probe_n)
    _, fi_tr, fa_tr, _ = predict(model, sub)
    for name, ftr, fte in (("img", fi_tr, fi_te), ("aud", fa_tr, fa_te)):
        if ftr is None:
            continue
        rec[f"probe_{name}"] = linear_probe(ftr, sub["y"], fte, te["y"])
        rec[f"featstd_{name}"] = fte.std(0).mean().item()
    return rec


# ----------------------------------------------------------------------------- run
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    name = f"{args.model}_{args.opt}_e{args.epochs}_s{args.seed}"
    out_dir = os.path.join(args.out, name)
    if os.path.exists(os.path.join(out_dir, "final.json")) and not args.force:
        print(f"SKIP {name}: final.json exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(args.device)
    seed_all(args.seed)
    tr, va, te, dstats = load_avmnist(args.data, dev, args.expect_train, args.expect_test)
    seed_all(args.seed)
    model = AVNet(args.model).to(dev)
    opt, sched = make_opt(model, args.opt)
    n_params = sum(p.numel() for p in model.parameters())
    meta = {"name": name, "args": vars(args), "data": dstats, "n_params": n_params,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    curve_path = os.path.join(out_dir, "curve.jsonl")
    cf = open(curve_path, "w")
    st = {"alpha": args.alpha, "mod_start": 0, "mod_end": 50, "epoch": 0,
          "gb_w": {"fused": 1 / 3, "img": 1 / 3, "aud": 1 / 3}}
    gb_sub = subset(tr, N_VAL)
    if args.model == "gblend":
        st["gb_prev"] = gb_losses(model, gb_sub, va)
    g = torch.Generator().manual_seed(args.seed)
    n_tr = len(tr["y"])
    print(f"START {name} params={n_params} n_train={n_tr} device={meta['device_name']}", flush=True)

    def log_eval(ep, extra):
        t_ev = time.time()
        rec = {"epoch": ep, **evaluate(model, tr, va, te, args.probe_n), **extra}
        rec["eval_sec"] = time.time() - t_ev
        cf.write(json.dumps(rec) + "\n")
        cf.flush()
        return rec

    rec = log_eval(0, {"gb_w": dict(st["gb_w"])} if args.model == "gblend" else {})
    print(f"ep 0 test {rec['test_acc']:.4f}", flush=True)
    for ep in range(1, args.epochs + 1):
        st["epoch"] = ep - 1  # official OGM-GE counts epochs from 0
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        perm = torch.randperm(n_tr, generator=g).to(dev)
        agg, nb = {}, 0
        for s in range(0, n_tr, args.batch):
            img, aud, y = get_batch(tr, perm[s:s + args.batch])
            r = train_step(model, opt, img, aud, y, st)
            for k, v in r.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
            if not math.isfinite(r["loss"]):
                print(f"NAN at epoch {ep} step {nb}", flush=True)
                cf.close()
                json.dump({"status": "nan", "epoch": ep}, open(os.path.join(out_dir, "final.json"), "w"))
                print(f"DONE {name} status=nan", flush=True)
                return
        if sched is not None:
            sched.step()
        extra = {f"train_{k}": v / nb for k, v in agg.items()}
        extra["epoch_sec"] = time.time() - t0
        if dev.type == "cuda":
            extra["max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        if args.model == "gblend" and ep % args.gb_super == 0:
            cur = gb_losses(model, gb_sub, va)
            st["gb_w"], raw = gb_weights(st["gb_prev"], cur)
            st["gb_prev"] = cur
            extra["gb_raw"] = raw
        if args.model == "gblend":
            extra["gb_w"] = dict(st["gb_w"])
        if ep % args.eval_every == 0 or ep == args.epochs:
            rec = log_eval(ep, extra)
            msg = " ".join(f"{k} {rec[k]:.4f}" for k in ("test_acc", "probe_img", "probe_aud") if k in rec)
            print(f"ep {ep} loss {extra['train_loss']:.4f} {msg} {extra['epoch_sec']:.1f}s", flush=True)
    cf.close()
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    json.dump({"status": "ok", **rec}, open(os.path.join(out_dir, "final.json"), "w"), indent=1)
    print(f"DONE {name} status=ok test_acc={rec['test_acc']:.4f}", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=KINDS)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--opt", default="adam", choices=["adam", "sgd_official"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.1, help="OGM-GE alpha (official default)")
    p.add_argument("--gb-super", type=int, default=5, help="Gradient Blending super-epoch length")
    p.add_argument("--probe-n", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--data", default=os.path.expanduser("~/MM_Wave1/data/avmnist"))
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/runs/A"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-train", type=int, default=60000)
    p.add_argument("--expect-test", type=int, default=10000)
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

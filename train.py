import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--lo", type=float, default=1.0)
p.add_argument("--hi", type=float, default=20.0)
p.add_argument("--samples", type=int, default=20000)
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--lr-min", type=float, default=1e-5)
p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
p.add_argument("--log", action="store_true",
               help="learn in log space: log(x1*x2) = log(x1) + log(x2)")
p.add_argument("--resume", action="store_true", help="continue from --model")
p.add_argument("--model", default="model.npz")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- resume?
if args.resume:
    m = np.load(args.model)
    n_layers = int(m["n_layers"])
    W = [m[f"W{i}"].copy() for i in range(n_layers)]
    b = [m[f"b{i}"].copy() for i in range(n_layers)]
    x_mean, x_std = m["x_mean"], m["x_std"]
    y_mean, y_std = float(m["y_mean"]), float(m["y_std"])
    use_log = bool(m["use_log"])
    print(f"Resuming from {args.model} (log mode: {use_log})")
else:
    use_log = args.log
    assert not use_log or args.lo > 0, "--log needs --lo > 0"

fx = np.log if use_log else (lambda v: v)   # input transform
fy = np.log if use_log else (lambda v: v)   # target transform


# ---------------------------------------------------------------- data
def make_data(n):
    X = rng.uniform(args.lo, args.hi, size=(n, 2))
    X[: n // 2] = np.round(X[: n // 2])      # half integer pairs, half decimals
    X = np.clip(X, args.lo, args.hi)
    return X, (X[:, 0] * X[:, 1]).reshape(-1, 1)


X_tr, y_tr = make_data(args.samples)
X_va, y_va = make_data(max(args.samples // 10, 500))

if not args.resume:
    x_mean, x_std = fx(X_tr).mean(0), fx(X_tr).std(0)
    y_mean, y_std = float(fy(y_tr).mean()), float(fy(y_tr).std())
    sizes = [2] + args.hidden + [1]
    W = [rng.standard_normal((a, c)) * np.sqrt(2 / a)          # He init
         for a, c in zip(sizes[:-1], sizes[1:])]
    b = [np.zeros((1, s)) for s in sizes[1:]]

Xn = (fx(X_tr) - x_mean) / x_std
yn = (fy(y_tr) - y_mean) / y_std
Xvn = (fx(X_va) - x_mean) / x_std
yvn = (fy(y_va) - y_mean) / y_std


# ---------------------------------------------------------------- network
def forward(X):
    acts, zs = [X], []
    for i in range(len(W)):
        z = acts[-1] @ W[i] + b[i]
        zs.append(z)
        acts.append(z if i == len(W) - 1 else np.maximum(0, z))   # ReLU
    return acts, zs


def predict_raw(X):
    out = forward((fx(X) - x_mean) / x_std)[0][-1] * y_std + y_mean
    return np.exp(out) if use_log else out


# ---------------------------------------------------------------- Adam
mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
mb = [np.zeros_like(x) for x in b]; vb = [np.zeros_like(x) for x in b]
b1, b2, eps = 0.9, 0.999, 1e-8
t = 0

best_val, best_W, best_b = np.inf, None, None

for epoch in range(args.epochs):
    # cosine learning-rate decay: big steps early, tiny steps at the end
    lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
        1 + np.cos(np.pi * epoch / args.epochs))

    idx = rng.permutation(len(Xn))
    epoch_loss = 0.0
    for s in range(0, len(idx), args.batch):
        j = idx[s:s + args.batch]
        acts, zs = forward(Xn[j])
        diff = acts[-1] - yn[j]
        epoch_loss += np.sum(diff ** 2)
        delta = 2 * diff / len(j)
        t += 1
        for i in reversed(range(len(W))):
            gW = acts[i].T @ delta
            gb = delta.sum(0, keepdims=True)
            if i > 0:
                delta = (delta @ W[i].T) * (zs[i - 1] > 0)
            mW[i] = b1 * mW[i] + (1 - b1) * gW
            vW[i] = b2 * vW[i] + (1 - b2) * gW ** 2
            mb[i] = b1 * mb[i] + (1 - b1) * gb
            vb[i] = b2 * vb[i] + (1 - b2) * gb ** 2
            W[i] -= lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
            b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)

    # validation (normalized MSE) and keep the best weights seen
    val_loss = np.mean((forward(Xvn)[0][-1] - yvn) ** 2)
    if val_loss < best_val:
        best_val = val_loss
        best_W = [w.copy() for w in W]
        best_b = [x.copy() for x in b]

    if epoch % 20 == 0 or epoch == args.epochs - 1:
        mae = np.mean(np.abs(predict_raw(X_va) - y_va))
        print(f"epoch {epoch:4d} | lr {lr:.5f} | train {epoch_loss / len(Xn):.6f}"
              f" | val {val_loss:.6f} | val MAE {mae:.3f}")

# restore the best weights and save
for i in range(len(W)):
    W[i][...] = best_W[i]
    b[i][...] = best_b[i]

mae = np.mean(np.abs(predict_raw(X_va) - y_va))
rel = np.mean(np.abs(predict_raw(X_va) - y_va) / y_va) * 100
print(f"\nBest model: val MAE {mae:.3f} | mean relative error {rel:.2f}%")

save = {f"W{i}": W[i] for i in range(len(W))}
save.update({f"b{i}": b[i] for i in range(len(b))})
np.savez(args.model, n_layers=len(W), x_mean=x_mean, x_std=x_std,
         y_mean=y_mean, y_std=y_std, use_log=use_log,
         lo=args.lo, hi=args.hi, **save)
print(f"Saved to {args.model}")
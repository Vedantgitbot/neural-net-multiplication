import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("pair", nargs="*", type=float, help="two numbers to multiply")
p.add_argument("--model", default="model.npz")
p.add_argument("--interactive", action="store_true")
args = p.parse_args()

m = np.load(args.model)
n_layers = int(m["n_layers"])
W = [m[f"W{i}"] for i in range(n_layers)]
b = [m[f"b{i}"] for i in range(n_layers)]
x_mean, x_std = m["x_mean"], m["x_std"]
y_mean, y_std = float(m["y_mean"]), float(m["y_std"])
use_log = bool(m["use_log"])
lo, hi = float(m["lo"]), float(m["hi"])


def predict(X):
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    h = ((np.log(X) if use_log else X) - x_mean) / x_std
    for i in range(n_layers):
        h = h @ W[i] + b[i]
        if i < n_layers - 1:
            h = np.maximum(0, h)
    out = h * y_std + y_mean
    return (np.exp(out) if use_log else out).ravel()


def show(a, c):
    pred = predict([[a, c]])[0]
    inside = lo <= a <= hi and lo <= c <= hi
    note = "" if inside else "  (outside training range)"
    print(f"{a:g} x {c:g} = {a * c:g} | predicted {pred:.2f}"
          f" (rounds to {round(pred)}){note}")


if len(args.pair) == 2:
    show(*args.pair)
elif args.interactive:
    while True:
        try:
            a, c = map(float, input("two numbers: ").split())
        except (ValueError, EOFError):
            break
        show(a, c)
else:
    rng = np.random.default_rng(123)
    print(f"Model trained on {lo:g}..{hi:g} (log mode: {use_log})\n")

    print("Spot checks:")
    for a, c in [(4, 6), (7, 8), (12, 13), (19, 20), (3, 3)]:
        show(a, c)

    def region_report(title, X):
        y = X[:, 0] * X[:, 1]
        pred = predict(X)
        err = np.abs(pred - y)
        edges = np.quantile(y, [0, 0.25, 0.5, 0.75, 1.0])
        print(f"\n{title}")
        print(f"{'product range':>16} | {'MAE':>8} | {'rel. err':>8}")
        for k in range(4):
            sel = (y >= edges[k]) & (y <= edges[k + 1])
            print(f"{edges[k]:7.0f}-{edges[k + 1]:<8.0f} | {err[sel].mean():8.3f}"
                  f" | {100 * (err[sel] / y[sel]).mean():7.2f}%")
        print(f"{'overall':>16} | {err.mean():8.3f} | {100 * (err / y).mean():7.2f}%")

    inside = np.round(rng.uniform(lo, hi, size=(2000, 2)))
    inside = np.clip(inside, lo, hi)
    region_report("In-range (integers inside the training range)", inside)

    outside = rng.uniform(hi, hi * 1.5, size=(500, 2))
    region_report(f"Extrapolation ({hi:g}..{hi * 1.5:g}, never seen in training)", outside)
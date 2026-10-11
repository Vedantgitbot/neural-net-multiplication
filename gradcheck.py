import numpy as np

rng = np.random.default_rng(0)

sizes = [2, 5, 5, 1]
W = [rng.standard_normal((a, c)) * np.sqrt(2 / a) for a, c in zip(sizes[:-1], sizes[1:])]
b = [rng.standard_normal((1, s)) * 0.1 for s in sizes[1:]]   # non-zero so bias grads are tested too

X = rng.uniform(0.5, 2.0, size=(8, 2))
y = (X[:, 0] * X[:, 1]).reshape(-1, 1)


def forward(X):
    acts, zs = [X], []
    for i in range(len(W)):
        z = acts[-1] @ W[i] + b[i]
        zs.append(z)
        acts.append(z if i == len(W) - 1 else np.maximum(0, z))   # ReLU
    return acts, zs


def loss():
    return np.mean((forward(X)[0][-1] - y) ** 2)


def backprop():
    """Analytic gradients (same math as train.py)."""
    acts, zs = forward(X)
    delta = 2 * (acts[-1] - y) / len(X)
    gW, gb = [None] * len(W), [None] * len(W)
    for i in reversed(range(len(W))):
        gW[i] = acts[i].T @ delta
        gb[i] = delta.sum(0, keepdims=True)
        if i > 0:
            delta = (delta @ W[i].T) * (zs[i - 1] > 0)
    return gW, gb


def numerical(param, eps=1e-6):
    """Finite-difference gradient for one parameter array (nudged in place)."""
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        old = param[idx]
        param[idx] = old + eps
        up = loss()
        param[idx] = old - eps
        down = loss()
        param[idx] = old
        grad[idx] = (up - down) / (2 * eps)
    return grad


gW, gb = backprop()
worst = 0.0
print(f"{'parameter':>10} | {'max relative error':>18}")
for name, params, analytic in [("W", W, gW), ("b", b, gb)]:
    for i in range(len(params)):
        num = numerical(params[i])
        rel = np.abs(analytic[i] - num) / np.maximum(np.abs(analytic[i]) + np.abs(num), 1e-12)
        worst = max(worst, rel.max())
        print(f"{name + str(i + 1):>10} | {rel.max():18.2e}")

print()
print("PASS: backprop matches the numerical gradient" if worst < 1e-5
      else "FAIL: gradients disagree, check the backward pass")
# Teaching a neural network to multiply (from scratch, NumPy only)

A small neural network, written in plain NumPy with no PyTorch or TensorFlow, that learns to multiply two numbers. Forward pass, backpropagation and the Adam optimizer are all written by hand, so you can read every step of how a network learns.

It's a learning project, not a practical multiplier (`a * b` is exact and free). The point is to see what a network actually learns, and where it breaks.

## Quick start

```bash
pip install -r requirements.txt

python train.py            # train on 1..20, saves model.npz
python test.py             # spot checks + error report
python test.py 7 8         # try one pair
python gradcheck.py        # verify the backprop code numerically
```

Learn in log space (a much better result, but see the note below):

```bash
python train.py --log --model model_log.npz
python test.py --model model_log.npz
```

Keep training an existing model: `python train.py --resume`

Other options: `python train.py --help` (range, network size, epochs, learning rate, ...).

## Results

Trained on products of numbers in 1..20. "Extrapolation" means numbers in 20..30 that the network never saw.

| | Plain mode | Log mode (`--log`) |
|---|---|---|
| In-range error (MAE) | 0.050 (0.18%) | 0.004 (0.00%) |
| Extrapolation error (MAE) | 47.8 (7.20%) | 0.52 (0.08%) |

In plain mode the error grows the further you go from the training range:

| Product size (20..30 inputs) | MAE | Relative error |
|---|---|---|
| 414-552 | 17.2 | 3.34% |
| 552-618 | 36.5 | 6.20% |
| 618-690 | 54.4 | 8.27% |
| 690-889 | 83.4 | 10.99% |

Your numbers will differ slightly with the random seed.

## What this shows

**Fitting is not understanding.** Inside the training range, the plain network is very accurate, and every spot check rounds to the right integer. Outside it, the error drifts upward. A ReLU network is built from straight pieces, so beyond the data it keeps extending the last piece as a straight line, while multiplication keeps curving. The network learned the shape of the function where it saw data, not the rule.

**Representation matters.** In log mode the network learns `log(a) + log(b)`, because `log(a * b) = log(a) + log(b)`. Addition is a straight line, which a ReLU network handles and extends perfectly, so extrapolation error drops about 90x. The catch is that this gives the network the key trick, so it's no longer learning multiplication from scratch. Plain mode is the honest version, and log mode shows how much the right representation helps. Log mode only works for positive inputs.

## How it works

The network is 2 inputs, two hidden layers of 64 ReLU neurons, and 1 output (about 4,400 weights). Training repeats four steps:

1. **Forward pass.** Each layer computes `z = input @ W + b` and applies ReLU (`max(0, z)`). The output is the network's guess.
2. **Loss.** Mean squared error, `mean((prediction - target)^2)`, measures how wrong the guess is.
3. **Backward pass.** The chain rule works backwards from the output to get the gradient of the loss for every weight. The error flows back only through neurons that were switched on, which is the `* (z > 0)` term (ReLU's derivative).
4. **Update.** Each weight moves a small step downhill. This project uses the Adam optimizer, which keeps momentum and gives each weight its own step size.

Things that make it train well:

- **Normalizing** inputs and targets, so the numbers going into ReLU stay near zero.
- **He initialization**, which sizes the starting weights so signals don't shrink or blow up through the layers.
- **Mini-batches** (128 samples per step) instead of the whole dataset.
- **Cosine learning-rate decay**: big steps early, tiny steps at the end.
- **Keeping the best weights** by validation score, so a bad late epoch can't ruin the saved model.

`gradcheck.py` compares the analytic gradients from backprop with a numerical estimate `(loss(w + eps) - loss(w - eps)) / (2 * eps)`. If they match, the backward pass is correct.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains the network and saves `model.npz` (supports `--resume` and `--log`) |
| `test.py` | Loads a saved model: spot checks, error by product size, extrapolation test |
| `gradcheck.py` | Verifies backprop against numerical gradients |

## Ideas to try

- **Square activations.** `a * b = ((a+b)^2 - (a-b)^2) / 4`, so a network with a squaring activation can represent multiplication exactly with a couple of neurons, and it should extrapolate. Does it?
- Change `--hidden` and `--hi` and watch how the error and the extrapolation cliff move.
- Learn other functions (`a + b`, `a ** 2`, `sin(a)`) and compare how hard each is.
- Smoother training: gradient clipping, learning-rate warmup, weight decay.
- Support zero and negative numbers (log mode can't).

## A note on how it was made

This project was built with AI assistance and then run, checked and understood by the author. The aim is learning, so read the code, change things and break it.

## License

MIT, see [LICENSE](LICENSE).
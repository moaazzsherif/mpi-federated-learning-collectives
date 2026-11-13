"""SCx61 Week 3 - shared helpers for the federated-averaging (FedAvg) lab.

Everything here is plain NumPy. No MPI calls live in this file, so the
serial oracle and the MPI programs use EXACTLY the same data and the same
model arithmetic.

Layout of the experiment
    rank 0      = server / aggregator   (holds NO data: shard size 0)
    ranks 1..5  = clients               (each owns a private, unequal shard)

Model: binary logistic regression with D = 20 features + a bias term, so
the weight vector has D + 1 = 21 float64 values (168 bytes).
"""
import faulthandler
import json
import os
import signal
import sys

import numpy as np

# ---------------------------------------------------------------- settings
D = 20                       # features
W_LEN = D + 1                # weights: 20 feature weights + 1 bias
SEED = 42
ROUNDS = 10                  # communication rounds
EPOCHS = 3                   # local full-batch gradient-descent steps per round
LR = 0.5                     # learning rate
COUNTS = [0, 800, 600, 400, 200, 100]   # rows per rank; rank 0 owns none
N_CLIENTS = len(COUNTS) - 1
N_TEST = 1000                # held-out test rows (kept at the server)

RESULTS_DIR = os.path.join(os.path.expanduser("~"), "work", "results")


def results_path(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


# -------------------------------------------------------------------- data
def make_data(seed=SEED):
    """Return (shards, test).

    shards[r] is an array of shape (COUNTS[r], D + 1); the LAST column is
    the 0/1 label. shards[0] has shape (0, D + 1). The data is non-IID:
    every client has its own feature offset and its own class balance.
    """
    rng = np.random.default_rng(seed)
    w_true = rng.normal(0.0, 1.0, D)
    offsets = rng.normal(0.0, 0.6, (N_CLIENTS, D))
    skews = [-1.5, -0.7, 0.0, 0.7, 1.5]     # shifts each client's class balance

    def draw(k, n):
        x = rng.normal(0.0, 1.0, (n, D)) + offsets[k]
        p = 1.0 / (1.0 + np.exp(-(x @ w_true + skews[k])))
        y = (rng.random(n) < p).astype(np.float64)
        return np.hstack([x, y[:, None]])

    shards = [np.empty((0, D + 1))]
    for k in range(N_CLIENTS):
        shards.append(draw(k, COUNTS[k + 1]))
    per = N_TEST // N_CLIENTS
    test = np.vstack([draw(k, per) for k in range(N_CLIENTS)])
    return shards, test


# ------------------------------------------------------------------- model
def initial_weights():
    """The server's starting model (deterministic, deliberately a poor guess)."""
    return np.random.default_rng(SEED + 7).normal(0.0, 1.0, W_LEN)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(w, x):
    return _sigmoid(x @ w[:D] + w[D])


def loss(w, data):
    """Mean cross-entropy on rows shaped (n, D + 1)."""
    x, y = data[:, :D], data[:, D]
    p = np.clip(predict_proba(w, x), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(w, data):
    x, y = data[:, :D], data[:, D]
    return float(np.mean((predict_proba(w, x) >= 0.5) == (y == 1.0)))


def grad(w, data):
    x, y = data[:, :D], data[:, D]
    err = predict_proba(w, x) - y
    g = np.empty(W_LEN)
    g[:D] = x.T @ err / len(y)
    g[D] = err.mean()
    return g


def local_train(w, data, epochs=EPOCHS, lr=LR):
    """Full-batch gradient descent on one client's shard (deterministic)."""
    w = np.array(w, dtype=np.float64, copy=True)
    for _ in range(epochs):
        w -= lr * grad(w, data)
    return w


# ---------------------------------------------------------------- schedule
def load_schedule(path):
    """Read a participation schedule.

    Returns {round_number: [selected client ranks]} for rounds 1..ROUNDS.
    A missing/None path means every client takes part in every round.
    """
    if path is None:
        return {r: list(range(1, N_CLIENTS + 1)) for r in range(1, ROUNDS + 1)}
    with open(path) as f:
        raw = json.load(f)
    return {int(r): list(v) for r, v in raw["selected"].items()}


# --------------------------------------------------------- logging / debug
def log(rank, rnd, msg):
    """One flushed line per event, so a stalled run leaves clear evidence."""
    print(f"rank {rank} round {rnd}: {msg}", flush=True)


_dump_file = None   # kept alive so faulthandler keeps a valid file object


def enable_stack_dump(rank):
    """Dump this process's Python stack to results/stack_rank<r>.txt on SIGUSR1."""
    global _dump_file
    _dump_file = open(results_path(f"stack_rank{rank}.txt"), "w")
    faulthandler.register(signal.SIGUSR1, file=_dump_file, all_threads=True)
    print(f"rank {rank}: pid {os.getpid()} on {os.uname().nodename}", flush=True)


def parse_arg(flag, default=None):
    """Tiny argv helper: returns the value after `flag`, or default."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

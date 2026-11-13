"""FedAvg with Allreduce - STARTER (fill in the 2 TODOs).

Every rank ends each round holding the new global model, so no Bcast is
needed at the start of the next round. Gather collects per-client losses.
"""
import numpy as np
from mpi4py import MPI

import fl_common as fl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
fl.enable_stack_dump(rank)
out_name = fl.parse_arg("--out", "mpi_v3.npz")

counts_rows = np.array(fl.COUNTS)
counts = counts_rows * (fl.D + 1)
displs = np.concatenate(([0], np.cumsum(counts)[:-1]))
if rank == 0:
    shards, test = fl.make_data()
    sendbuf = np.vstack(shards)
else:
    sendbuf = None
shard = np.empty((counts_rows[rank], fl.D + 1))
comm.Scatterv([sendbuf, counts, displs, MPI.DOUBLE], shard, root=0)

# Round 0 only: the server shares the starting model (everyone needs the same w0).
w = fl.initial_weights() if rank == 0 else np.zeros(fl.W_LEN)
comm.Bcast(w, root=0)

total = np.zeros(fl.W_LEN + 1)
W, ACC, LOSS = [], [], []
for rnd in range(1, fl.ROUNDS + 1):
    if rank == 0:
        contrib = np.zeros(fl.W_LEN + 1)
        my_loss = np.zeros(1)
    else:
        wk = fl.local_train(w, shard)
        n = shard.shape[0]
        contrib = np.append(n * wk, n)
        my_loss = np.array([fl.loss(wk, shard)])

    fl.log(rank, rnd, "entering Allreduce")
    comm.Allreduce(contrib, total, op=MPI.SUM)
    w = total[:-1] / total[-1]                            # every rank computes the average

    losses = np.empty(size) if rank == 0 else None
    comm.Gather(my_loss, losses, root=0)

    if rank == 0:
        W.append(w.copy())
        ACC.append(fl.accuracy(w, test))
        LOSS.append(fl.loss(w, test))
        print(f"round {rnd}: test acc={ACC[-1]:.3f} loss={LOSS[-1]:.4f} "
              f"client losses={np.round(losses[1:], 3).tolist()}", flush=True)
        np.savez(fl.results_path(out_name), W=np.array(W), acc=np.array(ACC), loss=np.array(LOSS))
    fl.log(rank, rnd, "done")

# --- do all ranks hold the SAME model? (measured, not assumed) -----------
all_w = comm.gather(w, root=0)       # lowercase = pickle; fine for a one-off check
if rank == 0:
    spread = max(float(np.max(np.abs(x - all_w[0]))) for x in all_w)
    bit_identical = all(np.array_equal(x, all_w[0]) for x in all_w)
    print(f"cross-rank check: max difference between ranks' final w = {spread:.3e}; "
          f"bit-identical = {bit_identical}", flush=True)

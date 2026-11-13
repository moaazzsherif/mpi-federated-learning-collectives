"""FedAvg with MPI collectives - STARTER (fill in the 3 TODOs).

Run:  mpirun --hostfile ~/work/hostfile -n 6 --tag-output python3 ~/work/fedavg_v2_collectives.py
"""
import numpy as np
from mpi4py import MPI

import fl_common as fl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
fl.enable_stack_dump(rank)
out_name = fl.parse_arg("--out", "mpi_v2.npz")

# ---- Step 1: hand out unequal shards with Scatterv -------------------
counts_rows = np.array(fl.COUNTS)                 # rows per rank (rank 0 -> 0)
counts = counts_rows * (fl.D + 1)                 # Scatterv counts are in ELEMENTS
displs = np.concatenate(([0], np.cumsum(counts)[:-1]))
if rank == 0:
    shards, test = fl.make_data()
    sendbuf = np.vstack(shards)                   # rank 0's block is empty
else:
    sendbuf = None
shard = np.empty((counts_rows[rank], fl.D + 1))   # rank 0: shape (0, 21) - legal
comm.Scatterv([sendbuf, counts, displs, MPI.DOUBLE], shard, root=0)
print(f"rank {rank}: shard rows = {shard.shape[0]}", flush=True)

# ---- Steps 2: FedAvg rounds -------------------------------------------
w = fl.initial_weights() if rank == 0 else np.zeros(fl.W_LEN)   # only the server knows w0
total = np.zeros(fl.W_LEN + 1)
W, ACC, LOSS = [], [], []
for rnd in range(1, fl.ROUNDS + 1):
    fl.log(rank, rnd, "entering Bcast")
    comm.Bcast(w, root=0)

    if rank == 0:
        contrib = np.zeros(fl.W_LEN + 1)          # server contributes nothing
    else:
        wk = fl.local_train(w, shard)
        n = shard.shape[0]
        contrib = np.append(n * wk, n)            # [n_k * w_k , n_k]

    fl.log(rank, rnd, "entering Reduce")
    comm.Reduce(contrib, total, op=MPI.SUM, root=0)

    if rank == 0:
        w[:] = total[:-1] / total[-1]
        W.append(w.copy())
        ACC.append(fl.accuracy(w, test))
        LOSS.append(fl.loss(w, test))
        print(f"round {rnd}: test acc={ACC[-1]:.3f} loss={LOSS[-1]:.4f}", flush=True)
        np.savez(fl.results_path(out_name), W=np.array(W), acc=np.array(ACC), loss=np.array(LOSS))
    fl.log(rank, rnd, "done")

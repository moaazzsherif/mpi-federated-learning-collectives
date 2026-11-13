"""FedAvg the Week-2 way: hand-written Send/Recv loops (for CONTRAST only).

Run:  mpirun --hostfile ~/work/hostfile -n 6 --tag-output python3 ~/work/fedavg_v1_manual.py

Count the MPI calls the server (rank 0) makes per round. Today's lab
replaces all of this with collectives.
"""
import numpy as np
from mpi4py import MPI

import fl_common as fl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
fl.enable_stack_dump(rank)
out_name = fl.parse_arg("--out", "mpi_v1.npz")

# --- hand out the shards (one Send per client) --------------------------
if rank == 0:
    shards, test = fl.make_data()
    for dest in range(1, size):
        comm.Send(np.ascontiguousarray(shards[dest]), dest=dest, tag=1)
    shard = shards[0]
else:
    shard = np.empty((fl.COUNTS[rank], fl.D + 1))
    comm.Recv(shard, source=0, tag=1)

w = fl.initial_weights() if rank == 0 else np.zeros(fl.W_LEN)
W, ACC, LOSS = [], [], []
for rnd in range(1, fl.ROUNDS + 1):
    calls = 0
    if rank == 0:
        for dest in range(1, size):                    # 5 Sends: the model
            comm.Send(w, dest=dest, tag=10)
            calls += 1
        total = np.zeros(fl.W_LEN + 1)
        buf = np.empty(fl.W_LEN + 1)
        for src in range(1, size):                     # 5 Recvs: the updates
            comm.Recv(buf, source=src, tag=20)
            total += buf
            calls += 1
        w = total[:-1] / total[-1]
        W.append(w.copy())
        ACC.append(fl.accuracy(w, test))
        LOSS.append(fl.loss(w, test))
        print(f"round {rnd}: test acc={ACC[-1]:.3f} loss={LOSS[-1]:.4f} "
              f"(server made {calls} Send/Recv calls this round)", flush=True)
        np.savez(fl.results_path(out_name), W=np.array(W), acc=np.array(ACC), loss=np.array(LOSS))
    else:
        comm.Recv(w, source=0, tag=10)
        wk = fl.local_train(w, shard)
        n = shard.shape[0]
        comm.Send(np.append(n * wk, n), dest=0, tag=20)

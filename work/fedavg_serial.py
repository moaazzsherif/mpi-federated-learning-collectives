"""Serial FedAvg oracle: simulates the 5 clients in ONE process.

Same data, same model, same arithmetic as the MPI programs - but no MPI.
It is the "ground truth" you compare your parallel runs against.

    python3 fedavg_serial.py                       # every client, every round
    python3 fedavg_serial.py --mask FILE.json      # only the clients the schedule selects
    python3 fedavg_serial.py --out NAME.npz        # output name (default serial.npz)

Writes ~/work/results/<out> with W (rounds x 21), acc and loss per round.
Aggregation rule each round:  w = sum(n_k * w_k) / sum(n_k)  over the
clients that take part (a client that is not selected has n_k = 0 that round).
"""
import numpy as np

import fl_common as fl

mask_path = fl.parse_arg("--mask")
out_name = fl.parse_arg("--out", "serial.npz")
schedule = fl.load_schedule(mask_path)

shards, test = fl.make_data()
w = fl.initial_weights()
W, ACC, LOSS = [], [], []
for rnd in range(1, fl.ROUNDS + 1):
    total = np.zeros(fl.W_LEN + 1)
    for k in schedule[rnd]:
        wk = fl.local_train(w, shards[k])
        n = shards[k].shape[0]
        total += np.append(n * wk, n)
    w = total[:-1] / total[-1]
    W.append(w.copy())
    ACC.append(fl.accuracy(w, test))
    LOSS.append(fl.loss(w, test))
    print(f"serial round {rnd}: participants={schedule[rnd]} "
          f"acc={ACC[-1]:.3f} loss={LOSS[-1]:.4f}")

np.savez(fl.results_path(out_name), W=np.array(W), acc=np.array(ACC), loss=np.array(LOSS))
print("wrote", fl.results_path(out_name))

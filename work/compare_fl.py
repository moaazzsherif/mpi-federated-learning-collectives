"""Compare an MPI run with the serial oracle, round by round.

    python3 compare_fl.py                                  # mpi_v2.npz vs serial.npz
    python3 compare_fl.py --mpi mpi_v3.npz                 # another MPI result
    python3 compare_fl.py --mpi mpi_sampling.npz --serial serial_masked.npz
    python3 compare_fl.py --tol 1e-12                      # change the tolerance

MPI sums contributions in an order the library chooses, floating-point
addition is not associative, so we compare with a tolerance, not ==.
TOL below was chosen from measurements (see the lab README).
"""
import sys

import numpy as np

import fl_common as fl

TOL = 1e-12

mpi_name = fl.parse_arg("--mpi", "mpi_v2.npz")
ser_name = fl.parse_arg("--serial", "serial.npz")
tol = float(fl.parse_arg("--tol", TOL))

try:
    m = np.load(fl.results_path(mpi_name))["W"]
    s = np.load(fl.results_path(ser_name))["W"]
except FileNotFoundError as e:
    print("FAIL: missing result file:", e.filename)
    sys.exit(2)

print(f"MPI result    : {mpi_name}  ({len(m)} rounds)")
print(f"serial oracle : {ser_name}  ({len(s)} rounds)")
print(f"tolerance     : {tol:g}")
print(f"{'round':>5}  {'max|dw|':>12}")
first_bad = None
for r in range(min(len(m), len(s))):
    d = float(np.max(np.abs(m[r] - s[r])))
    flag = "" if d <= tol else "  <-- exceeds tolerance"
    if d > tol and first_bad is None:
        first_bad = r + 1
    print(f"{r + 1:>5}  {d:12.3e}{flag}")

if len(m) < len(s):
    print(f"FAIL: MPI run produced only {len(m)} of {len(s)} rounds")
    sys.exit(1)
if first_bad is not None:
    print(f"FAIL: first round beyond tolerance = {first_bad}")
    sys.exit(1)
print("PASS")

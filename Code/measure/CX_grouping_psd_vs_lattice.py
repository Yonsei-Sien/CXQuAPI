"""Is the lattice criterion actually weaker than PSD in practice, i.e. does it reach alpha = 1 with smaller groups?

PSD margin:     max s  s.t.  sum_G embed(X_G) = M,  X_G >= s I          (SDP; alpha = 1 guaranteed if s >= 0)
lattice alpha:  the LP of cx_grouping_lp (exact alpha on the blip lattice)
A row where the margin is negative while the lattice alpha is 1 is a grouping the semidefinite criterion rejects
and the lattice criterion accepts: the paper's Table S2, and the reason the criterion must be the lattice one.
Runs in the isolated cvxpy venv (~/.venvs/cvx), which also sees the base site-packages.
  [Output] Data/<SYSTEM>/Grouping/psd_vs_lattice.json : one row per (N, g), with the CNOT count of that family
           stdout: the same, as a table
"""
import json
import os
import sys

import cvxpy as cp
import numpy as np

from lib.cx_grouping_lp import optimal_grouping, pairs_family, ucry_cnot, window_family
from measure.CX_psd_hierarchy_probe import M_matrix, STRONG
from measure.CX_grouping_lp_scan import LATTICE_TLS
from lib.paths import data_path

SYSTEM   = 'TLS_strong'      # the driver in Running/ sets this; data lands in Data/<SYSTEM>/Grouping/
PROPERTY = 'Grouping'
FILENAME = 'psd_vs_lattice.json'   # one file per bath, since a driver may scan the same system twice
CASE     = STRONG            # the physical system whose M matrix is scanned
LATTICE  = LATTICE_TLS       # blip values of a two-state system
N_Q      = 1                 # path qubits per time slice, which fixes the CNOT count of a group
N_RANGE  = range(2, 8)
MAX_ROWS = 4_000_000         # inequality cap, checked BEFORE a family is built: |LATTICE|^g per group


# Inequality rows the linear program will carry at this group size, one per +-pair of lattice vectors.
# Predicted rather than measured, since |LATTICE|^g is an out-of-memory on a careless g.
#   [Input]  groups : list of tuples of time indices
#   [Output] rows : total inequalities
def program_rows(groups):
    zero = 1 if np.any(LATTICE == 0.0) else 0
    return sum((len(LATTICE) ** len(G) - zero) // 2 for G in groups)


# max s subject to sum_G X_G = M and X_G >= s I: the best worst-case eigenvalue any split can hold.
# s >= 0 certifies alpha = 1 under the semidefinite criterion; s < 0 means no split of this family is PSD.
def psd_margin(M, groups):
    n = len(M)
    X = [cp.Variable((len(G), len(G)), symmetric=True) for G in groups]
    s = cp.Variable()
    total = 0
    for G, XG in zip(groups, X):
        P = np.zeros((n, len(G)))
        P[list(G), range(len(G))] = 1.0
        total = total + P @ XG @ P.T
    cons = [total == M] + [XG - s * np.eye(len(G)) >> 0 for G, XG in zip(groups, X)]
    cp.Problem(cp.Maximize(s), cons).solve(solver=cp.CLARABEL)
    return float(s.value)


def main():
    rows = []
    out = os.path.join(data_path(SYSTEM, PROPERTY), FILENAME)
    print(f"{'N':>2} {'g':>2} {'CNOT':>8}  {'PSD margin':>11}  {'lattice alpha':>13}   verdict", flush=True)
    for N in N_RANGE:
        M = M_matrix(CASE, N)
        n = N + 1
        for g in range(2, n + 1):
            fam = pairs_family(n) if g == 2 else window_family(n, g)
            if program_rows(fam) > MAX_ROWS:
                print(f'{N:>2} {g:>2}  skipped: {program_rows(fam)} inequalities exceeds the cap', flush=True)
                break
            s = psd_margin(M, fam)
            a, _, _ = optimal_grouping(M, fam, LATTICE)
            cx = ucry_cnot(fam, N_Q)
            psd_ok, lat_ok = s >= -1e-9, a < 1 + 1e-9
            verdict = ('both' if psd_ok and lat_ok else
                       'lattice only' if lat_ok else 'neither')
            print(f"{N:>2} {g:>2} {cx:>8}  {s:>11.3e}  {a:>13.4f}   {verdict}", flush=True)
            rows.append(dict(system=CASE['name'], N=N, g=g, cx=cx, margin=s, alpha=a,
                             psd_ok=psd_ok, lattice_ok=lat_ok, verdict=verdict))
            json.dump(rows, open(out, 'w'), indent=1)
            if psd_ok and lat_ok:
                break
    print(f'\n{len(rows)} rows -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

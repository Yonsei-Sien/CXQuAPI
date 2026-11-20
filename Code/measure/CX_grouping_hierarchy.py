"""How small can the groups be at strong coupling while the construction stays exact?

For the strongly damped TLS the LP is run over two families of groups at every memory length:

    pairs      : {(0,1), (2,3), ...} plus the far pairs, the closed form the paper starts from
    g-windows  : {(0..g-1), (g..2g-1), ...}, windows of g consecutive time slices

and each family is scored by the exact alpha its LP reaches on the blip lattice and by the CNOT
count of the uniformly controlled rotations that realize it. alpha = 1 means the grouped influence
functional equals the full one; the smallest g reaching it is the hierarchy the paper's Figure 2b
shows, with the gate cost of buying it.

Restored from the Attic as CX_grouping_data.py (see Trials.md). Only the strong-coupling TLS part is
kept, since that is the file Analysis/Fig2.ipynb reads; the three-state beta scans it also wrote are
not read anywhere and were argv driven, so they are gone rather than silently half-run.

  [Input]  N_RANGE : memory lengths to scan; a group of g slices is built for g = 2..N+1
  [Output] Data/TLS_strong/Grouping/grouping_tls_strong.json : one row per (N, g) with
           (N, g, alpha, groups, cx_ucry, fused), fused marking g >= N+1 where one group holds
           the whole influence functional
"""
import json
import os

import numpy as np

from lib.cx_grouping_lp import optimal_grouping, pairs_family, ucry_cnot, window_family
from lib.paths import data_path
from measure.CX_psd_hierarchy_probe import M_matrix, STRONG

SYSTEM   = 'TLS_strong'
PROPERTY = 'Grouping'
CASE     = STRONG                          # xi = 1.2, w_c = 2.5, beta = 0.2: the strongly damped TLS
LATTICE  = np.array([-2.0, 0.0, 2.0])      # Delta = s+ - s- for s in {+1, -1}
N_Q      = 1                               # one path qubit per slice, which fixes the CNOT count
N_RANGE  = range(1, 8)


def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)


# alpha and gate cost of every group family, over the memory lengths of N_RANGE.
#   [Input]  M_of : N -> blip matrix of the influence functional over N + 1 slices
#   [Output] rows : one dict per (N, g), ending at the g that fuses the whole functional
def scan(M_of, lattice, n_q, Ns, g_max_of):
    rows = []
    for N in Ns:
        M, n = M_of(N), N + 1
        for g in range(2, g_max_of(N) + 1):
            # pairs is the closed form at g = 2; above it the family is windows of g slices
            fam = pairs_family(n) if g == 2 and n > 1 else window_family(n, g)
            a, _, _ = optimal_grouping(M, fam, lattice)
            rows.append(dict(N=N, g=min(g, n), alpha=a, groups=len(fam),
                             cx_ucry=ucry_cnot(fam, n_q), fused=g >= n))
            print(f'  N={N:>2} g={g}  alpha={a:.6g}  groups={len(fam)}  '
                  f'cx={ucry_cnot(fam, n_q)}', flush=True)
            if g >= n:                      # one group already holds every slice: nothing larger exists
                break
    return rows


def main():
    print(CASE['name'], flush=True)
    rows = scan(lambda N: M_matrix(CASE, N), LATTICE, N_Q, N_RANGE, lambda N: N + 1)
    out = os.path.join(ROOT_DIR(), 'grouping_tls_strong.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print('written:', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

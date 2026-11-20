"""Task: how small can the groups be at strong coupling while keeping alpha = 1?

For the TLS at weak and strong coupling, N = 1..7, compare the minimal alpha reachable by
  pairs (g = 2)  -- current closed form, and the LP optimum over the same family
  windows of g consecutive times + remaining far pairs, g = 3..N+1   (LP optimum)
with the IF-circuit CNOT count of each family.  alpha is exact: the LP optimizes over the blip lattice itself.
"""
import sys

import numpy as np

from lib.cx_grouping_lp import optimal_grouping, pairs_family, window_family, ucry_cnot
from measure.CX_psd_hierarchy_probe import M_matrix, WEAK, STRONG

LATTICE_TLS = np.array([-2.0, 0.0, 2.0])     # Delta = s+ - s- with s in {+1, -1}


def main():
    for system in (WEAK, STRONG):
        print(f"\n=== {system['name']} ===   alpha_min (LP)  [CNOT of the IF groups]", flush=True)
        print(f"{'N':>2} {'pairs':>18} " + ' '.join(f"{'g=' + str(g):>18}" for g in range(3, 9)), flush=True)
        for N in range(1, 8):
            M = M_matrix(system, N)
            n = N + 1
            cells = []
            fam = pairs_family(n)
            a, _, _ = optimal_grouping(M, fam, LATTICE_TLS)
            cells.append(f"{a:>8.4f} [{ucry_cnot(fam, 1):>6d}]")
            for g in range(3, 9):
                if g > n:
                    cells.append(f"{'':>18}"); continue
                fam = window_family(n, g)
                a, _, _ = optimal_grouping(M, fam, LATTICE_TLS)
                cells.append(f"{a:>8.4f} [{ucry_cnot(fam, 1):>6d}]")
            print(f"{N:>2} " + ' '.join(cells), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

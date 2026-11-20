"""Does the group size the program needs settle down as the propagation grows, or keep growing with it?

One group of g slices costs 2^{2 g n_q} CNOTs, so a g* that keeps rising with N is an exponential whose exponent
itself rises: the construction would then have no regime in which the gate count is merely exponential.  The
influence coefficients decay with |k - k'|, so g* should saturate once the window covers the reach of the bath
correlation -- but at full memory M is never banded, and whether saturation actually happens is a question about
the numbers rather than about the argument.  This follows g*(N) for each system as far as the program stays
affordable.
  [Input]  argv: largest N (default 10), largest N for the three-state model (default 7)
  [Output] Data/Grouping/gstar_vs_N.json
"""
import json
import os
import sys
import time

import numpy as np

from lib.paths import data_path
from lib.cx_systems_lib import tls, sim2001, ohmic
from measure.CX_coupling_sweep import M_matrix, smallest_exact_group, LAT_TLS, LAT_3LS

PROPERTY = 'Grouping'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = ((tls(), LAT_TLS, 1, 'TLS'),
         (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), LAT_TLS, 1, 'TLS-strong'),
         (sim2001(0.72), LAT_3LS, 2, 'Sim2001'))


# Reach of the influence coefficients: the largest |k - k'| whose |Re eta| is above a fraction of the diagonal
def reach(M, frac=1e-3):
    n = len(M)
    scale = max(abs(M[k, k]) for k in range(n))
    far = [abs(k - j) for k in range(n) for j in range(n) if abs(M[k, j]) > frac * scale]
    return max(far) if far else 0


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    N_max3 = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    os.makedirs(ROOT_DIR(), exist_ok=True)   # data_path already does this; kept for symmetry
    rows = []
    for system, lattice, n_q, tag in CASES:
        top = N_max3 if tag == 'Sim2001' else N_max
        print(f'{tag}  (d={system["N_state"]})', flush=True)
        for N in range(2, top + 1):
            M = M_matrix(system, N)
            t0 = time.time()
            g, cx, a_pair, a_g = smallest_exact_group(M, lattice, n_q, N)
            rows.append(dict(system=tag, N=N, slices=len(M), g=int(g), cx=int(cx),
                             reach=int(reach(M)), alpha_pair=float(a_pair), seconds=time.time() - t0))
            print(f'  N={N:2d}  slices={len(M):2d}  g*={g}  reach={reach(M):2d}  CNOT={cx:<12d}'
                  f'  alpha(pairs)={a_pair:.3e}  [{time.time() - t0:.1f}s]', flush=True)
            json.dump(rows, open(os.path.join(ROOT_DIR(), 'gstar_vs_N.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

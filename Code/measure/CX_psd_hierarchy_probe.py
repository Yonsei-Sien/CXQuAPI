"""Which group size does the blip matrix M actually need for lambda_G = 1?

Three nested criteria, weakest requirement last:
  (a) diagonal dominance          M_kk >= sum_k' |M_kk'|          -- the current closed-form 2-group shares
  (b) scaled diagonal dominance   comparison matrix C(M) >= 0      -- 2-groups with Perron-vector shares
      (factor width 2: Boman, Chen, Parekh & Toledo, Linear Algebra Appl. 405, 239 (2005))
  (c) M >= 0 itself               always true at full memory       -- one group of all N+1 times (= fused)
and in between, windows of g consecutive times: M is split into a far part (2-groups, shares w) and a banded near
part of bandwidth g-1, whose Cholesky factor has no fill-in, so each column is a PSD rank-1 group on g times.

  [Output] per (system, N): lambda_min of M, of C(M), diag-dominance slack, and for g = 3..N+1 whether the
           banded remainder is PSD (i.e. whether g-time windows reach alpha = 1)
"""
import contextlib
import io
import sys

import numpy as np

from lib.cxquapi_psd_lib import CXQUAPI_PSD
from lib.cx_systems_lib import tls, ohmic

WEAK   = tls()
STRONG = dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2)


def M_matrix(system, N):
    obj = CXQUAPI_PSD(system['H_q'], system['S_q'], system['dt'], N, N_memory=N,
                      N_state=system['N_state'], beta=system['beta'])
    obj.spectral_density = system['sd']
    with contextlib.redirect_stdout(io.StringIO()):
        return obj.gen_M_matrix()


# C(M): same diagonal, off-diagonals replaced by -|M_kk'|. M is scaled-diagonally-dominant iff C(M) >= 0.
def comparison(M):
    C = -np.abs(M)
    np.fill_diagonal(C, np.diag(M))
    return C


# 2-group shares for a set of far couplings, scaled by a positive vector v (v = Perron vector of C is optimal)
#   w_kk' = |M_kk'| v_k' / (M_kk v_k)  -> det of each 2x2 block vanishes, and sum_k' w_kk' <= 1 iff C v >= 0
def far_shares(M, far, v):
    D = np.zeros(len(M))
    for k, kp in far:
        D[k]  += abs(M[k, kp]) * v[kp] / v[k]
        D[kp] += abs(M[k, kp]) * v[k] / v[kp]
    return D                                     # absolute diagonal mass the far 2-groups consume


def window_feasible(M, g):
    n = len(M)
    far  = [(k, kp) for k in range(n) for kp in range(k) if k - kp >= g]
    near = M.copy()
    for k, kp in far:
        near[k, kp] = near[kp, k] = 0.0
    if not far:
        return float(np.linalg.eigvalsh(near).min())
    # far part: Perron vector of its own comparison matrix restricted to far couplings + full diagonal
    Cf = -np.abs(M - near)
    np.fill_diagonal(Cf, np.diag(M))
    w, V = np.linalg.eigh(Cf)
    v = np.abs(V[:, 0])
    v = np.where(v > 1e-300, v, 1e-300)
    D = far_shares(M, far, v)
    near[np.diag_indices(n)] = np.diag(M) - D     # what is left of the diagonal for the banded windows
    return float(np.linalg.eigvalsh(near).min())


def main():
    for system in (WEAK, STRONG):
        print(f"\n=== {system['name']} ===", flush=True)
        print(f"{'N':>2} {'lmin M':>11} {'lmin C(M)':>11} {'DD slack':>11}   "
              + '  '.join(f'g={g}:lmin' for g in range(3, 8)), flush=True)
        for N in range(1, 8):
            M = M_matrix(system, N)
            n = N + 1
            dd = min(M[k, k] - sum(abs(M[k, j]) for j in range(n) if j != k) for k in range(n))
            row = [window_feasible(M, g) if g <= n else np.nan for g in range(3, 8)]
            print(f"{N:>2} {np.linalg.eigvalsh(M).min():>11.3e} {np.linalg.eigvalsh(comparison(M)).min():>11.3e} "
                  f"{dd:>11.3e}   " + '  '.join(f'{x:>10.2e}' if x == x else f"{'':>10}" for x in row),
                  flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

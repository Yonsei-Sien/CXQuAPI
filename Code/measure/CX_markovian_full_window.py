"""Why the single full window needs no linear program.

The grouping LP of CX_markovian_shape scans group sizes g = 2 ... N + 1 and stops at the first one reaching
alpha = 1. At the last size there is one group over every slice, so the equality constraint

    sum_G embed(X_G) = M        ->        X_G = M

leaves the program no freedom, and the only question is the sign of the generator it has to carry,

    max_{Delta in Lambda^(N+1)} -Delta^T M Delta.

If that is <= 0 then t_G = 0 and alpha = 1 with nothing to solve. This scan shows it is, in the strongest
available way: M is positive DEFINITE at every bath of the Table "One bath across the Markovian boundary"
grid, so -Delta^T M Delta < 0 for every real Delta, let alone for the lattice. The lattice maximum is
measured directly as well, at the two lengths where enumerating |Lambda|^(N+1) vectors is affordable, so
the conclusion does not rest on the eigenvalue alone.

  [Input]  the same grid as CX_markovian_shape: BETAS x STEPS_PER_TAU at DT, xi = 4.188/pi, N in N_R
  [Output] Data/Common/Bath/markovian_full_window.json : one row per (bath, length) with
             lmin    : smallest eigenvalue of the symmetrized M
             latmax  : max over Lambda^(N+1) of -Delta^T M Delta, or None where not enumerated
"""
import json
import os

import numpy as np

from lib.cx_systems_lib import sim2001, ohmic
from measure.CX_coupling_sweep import M_matrix, LAT_3LS
from measure.CX_markovian_shape import DT, BETAS, STEPS_PER_TAU
from lib.paths import data_path

PROPERTY = 'Bath'
SYSTEM = 'Common'
XI = 2 * 2.094 / np.pi     # the three-state model's coupling, as in CX_markovian_shape
LAT_MAX_UPTO = 7           # lengths at or below this get the full lattice enumeration, 5^(N+1) vectors
CHUNK = 1 << 18            # lattice vectors held at once, so the enumeration never builds 5^(N+1) rows


def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)


# max_{Delta in Lambda^n} -Delta^T M Delta, streamed: tile index -> base-|Lambda| digits -> one einsum
#   [Input]  M : (n, n) influence matrix;  lat : the blip lattice
#   [Output] the largest value the full-window generator takes on the lattice
def lattice_max(M, lat):
    n, nl, lat = len(M), len(lat), np.asarray(lat, dtype=float)
    pw, total, best = nl ** np.arange(n), nl ** len(M), -np.inf
    for lo in range(0, total, CHUNK):
        idx = np.arange(lo, min(lo + CHUNK, total))
        V = lat[(idx[:, None] // pw) % nl]
        best = max(best, float(-np.einsum('ij,jk,ik->i', V, M, V).max()))
    return best


def main(N_R=(5, 7, 10)):
    rows = []
    for beta in BETAS:
        for s_tau in STEPS_PER_TAU:
            # H and S never enter M; the bath alone fixes it, as in CX_markovian_shape
            bath = dict(sim2001(beta), dt=DT, beta=beta, sd=ohmic(XI, 1.0 / (s_tau * DT)))
            for N in N_R:
                M = M_matrix(bath, N)
                # symmetrized, since only the quadratic form enters and the program carries X = X^T
                lmin = float(np.linalg.eigvalsh((M + M.T) / 2).min())
                lm = lattice_max(M, LAT_3LS) if N <= LAT_MAX_UPTO else None
                rows.append(dict(beta=beta, steps_per_tau=float(s_tau), N=N, dt=DT, xi=XI,
                                 lmin=lmin, latmax=lm))
                print(f'beta={beta:>5.3f}  1/(w_c dt)={s_tau:>5.3f}  N={N:<3d} '
                      f'lmin={lmin:+.4e}' + ('' if lm is None else f'  latmax={lm:+.4e}'), flush=True)
    lo = min(r['lmin'] for r in rows)
    hi = max(r['lmin'] for r in rows)
    print(f'\nlambda_min(M) over the grid : [{lo:.4e}, {hi:.4e}]  -- M is positive definite everywhere')
    worst = max(r['latmax'] for r in rows if r['latmax'] is not None)
    print(f'largest full-window generator on the lattice, where enumerated : {worst:+.4e}')

    os.makedirs(ROOT_DIR(), exist_ok=True)
    out = os.path.join(ROOT_DIR(), 'markovian_full_window.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

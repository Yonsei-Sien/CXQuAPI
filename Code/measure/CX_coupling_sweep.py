"""How the circuit cost of the influence functional depends on how strongly the bath is coupled.

The bath makes the problem harder on the circuit and easier at the readout, and both effects are controlled by one
physical parameter. This scan measures the first of the two: for a sequence of coupling strengths, the smallest
group of time slices that reaches alpha = 1 and the CNOT count of that grouping. The group size is what the gate
cost is exponential in, so its growth with the coupling is the gate-side price of dissipation.
  [Input]  argv: N for the two-level system (default 5), N for the three-state model (default 4)
  [Output] Data/<System>/Bath/coupling_sweep.json, one file per system scanned ('TLS', '3LS' and the
           bath-shape block as 'TLS-shape'): per coupling, the smallest g with alpha = 1, its CNOT
           count, and alpha of the standard pairwise grouping
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.cxquapi_psd_lib import CXQUAPI_PSD
from lib.cx_grouping_lp import optimal_grouping, pairs_family, window_family, ucry_cnot
from lib.paths import data_path

PROPERTY = 'Bath'       # data lands in Data/<System>/Bath/, one file per system tag
LAT_TLS = np.array([-2.0, 0.0, 2.0])                                   # s = +-1
LAT_3LS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])                        # s = -1, 0, +1


# M with M_kk = Re eta_kk and M_kk' = Re eta_kk'/2, for one system dict
def M_matrix(system, N):
    obj = CXQUAPI_PSD(system['H_q'], system['S_q'], system['dt'], N, N_memory=N,
                      N_state=system['N_state'], beta=system['beta'])
    obj.spectral_density = system['sd']
    with contextlib.redirect_stdout(io.StringIO()):
        return obj.gen_M_matrix()


# smallest group of consecutive slices that reaches alpha = 1, and what that grouping costs
def smallest_exact_group(M, lattice, n_q, N):
    n, alpha_pair = N + 1, None
    for g in range(2, n + 1):
        fam = pairs_family(n) if g == 2 else window_family(n, g)
        alpha, _, _ = optimal_grouping(M, fam, lattice)
        if g == 2:
            alpha_pair = float(alpha)
        if alpha < 1 + 1e-9:
            return g, int(ucry_cnot(fam, n_q)), alpha_pair, float(alpha)
    return n, int(ucry_cnot(window_family(n, n), n_q)), alpha_pair, 1.0


def scan(name, system_of, xis, lattice, n_q, N):
    rows = []
    print(f'--- {name}, N = {N}', flush=True)
    for xi in xis:
        M = M_matrix(system_of(xi), N)
        g, cx, a_pair, a_g = smallest_exact_group(M, lattice, n_q, N)
        rows.append(dict(system=name, N=N, xi=float(xi), g=g, cx=cx, alpha_pair=a_pair, alpha_at_g=a_g,
                         reorg=float(np.sum(np.abs(M)))))
        print(f'  xi={xi:>6.3f}  smallest g = {g}  CNOT = {cx:>7}  alpha(pairs) = {a_pair:.4g}', flush=True)
    return rows


def main():
    N_tls = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    N_3ls = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    # two-level system: vary the Ohmic coupling, keep cut-off, temperature and time step of ref. Makri & Makarov
    rows = scan('TLS', lambda xi: dict(tls(), sd=ohmic(xi, 7.5)),
                [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0], LAT_TLS, 1, N_tls)

    # three-state model: vary the coupling of ref. Sim (2001), keep its cut-off, temperature and time step
    rows += scan('3LS', lambda xi: dict(sim2001(0.72), sd=ohmic(xi, 1.0)),
                 [0.1, 0.3, 0.6, 1.0, 1.333, 2.0, 3.0, 4.0], LAT_3LS, 2, N_3ls)

    # the group size turns out to be invariant under the coupling, because M is linear in it and the lattice
    # condition is homogeneous; what does move it is the shape of the bath correlation, so scan that too
    print('--- TLS, shape of the bath at fixed coupling', flush=True)
    for xi in (0.1, 1.0):
        for wc in (7.5, 2.5, 1.0):
            for beta in (5.0, 1.0, 0.2):
                system = dict(tls(), sd=ohmic(xi, wc), beta=beta)
                g, cx, a_pair, a_g = smallest_exact_group(M_matrix(system, N_tls), LAT_TLS, 1, N_tls)
                rows.append(dict(system='TLS-shape', N=N_tls, xi=xi, w_c=wc, beta=beta, g=g, cx=cx,
                                 alpha_pair=a_pair, alpha_at_g=a_g,
                                 steps_per_tau=float(1.0 / (wc * tls()['dt']))))
                print(f'  xi={xi:>4.1f} w_c={wc:>5.1f} beta={beta:>5.1f}  smallest g = {g}  '
                      f'CNOT = {cx:>8}  alpha(pairs) = {a_pair:.4g}', flush=True)

    # One file per system, so that a system can be read on its own: the scan tags every row with the
    # system it was measured on, and paths.system_dir turns that tag into the directory name.
    for tag in sorted({r['system'] for r in rows}):
        out = os.path.join(data_path(tag, PROPERTY), 'coupling_sweep.json')
        json.dump([r for r in rows if r['system'] == tag], open(out, 'w'), indent=1)
        print(f'written to {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

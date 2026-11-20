"""Circuit against the path sum built from the SAME influence coefficients.

CX_exactness_recheck.py compares the circuit with QUAPI.FPF, which recomputes eta by its own quadrature, so
its residual mixes the circuit's error with the reference's. This script compares the circuit against the
path sum of CX_mc_readout_N6.all_weights, which uses the very eta the circuit was built from, so whatever is
left is the circuit alone. FPF is reported beside it to show how much of the earlier number was the
reference.

  [Input]  argv: largest N to attempt (default 5)
  [Output] Data/<System>/Validation/exactness_direct.json
"""
import contextlib
import io
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import (CXQUAPI, alpha_generator, optimal_grouping, window_groups, h_bar)
from lib.cx_systems_lib import tls, sim2001, ohmic, run_fpf
from lib.CX_mc_readout_N6 import all_weights, normalize
from lib.paths import data_path

PROPERTY = 'Validation'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = (tls(), dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), sim2001(0.72))
LATTICE = {2: (-2.0, 0.0, 2.0), 3: (-2.0, -1.0, 0.0, 1.0, 2.0)}


def g_star(M, d):
    for g in range(2, len(M) + 1):
        fam = window_groups(len(M), g)
        if optimal_grouping(M, fam, LATTICE[d])[0] < 1.0 + 1e-9:
            return g, len(fam)
    return None, None


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = []
    print(f"{'system':>20}{'N':>3}{'g*':>4}{'qubits':>8}{'|circ-sum|':>12}{'|sum-FPF|':>12}{'|circ-FPF|':>12}")
    for system in CASES:
        d = system['N_state']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        for N in range(2, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            M = 0.5 * (np.real(np.asarray(eta)) + np.real(np.asarray(eta)).T)
            g, n_group = g_star(M, d)
            n_q = 2 * (N + 1) * (1 if d == 2 else 2) + (n_group or 0)
            if g is None or n_q > 24:
                print(f"{system['name']:>20}{N:>3}{str(g):>4}{n_q:>8}   skipped", flush=True)
                continue
            # the reference path sum, from the same eta the circuit carries
            cfg = np.array(list(itertools.product(range(d), repeat=2 * (N - 1))), dtype=int)
            ref = normalize(np.array([[all_weights(a, b, cfg, U, s, eta).sum() for b in range(d)]
                                      for a in range(d)]))
            with contextlib.redirect_stdout(io.StringIO()):
                obj = CXQUAPI(system['H_q'], system['S_q'], system['dt'], N, N_memory=N, N_state=d,
                              beta=system['beta'], IF='lp', IF_setting={'group_size': g})
                obj.spectral_density = system['sd']
                rho_q = np.asarray(obj.run())
            fpf = run_fpf(system, N)
            a = float(np.abs(rho_q - ref).max())
            b = float(np.abs(ref - fpf).max())
            c = float(np.abs(rho_q - fpf).max())
            rows.append(dict(system=system['name'], N=N, g=g, qubits=n_q,
                             circ_vs_sum=a, sum_vs_fpf=b, circ_vs_fpf=c))
            print(f"{system['name']:>20}{N:>3}{g:>4}{n_q:>8}{a:>12.3e}{b:>12.3e}{c:>12.3e}", flush=True)
    os.makedirs(ROOT_DIR(), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT_DIR(), 'exactness_direct.json'), 'w'), indent=1)
    if rows:
        print(f"\nworst circuit-vs-path-sum = {max(r['circ_vs_sum'] for r in rows):.3e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

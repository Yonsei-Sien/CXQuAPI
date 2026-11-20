"""How far in time on-the-fly filtering reaches, against the brute-force path sum it replaces.

The filter never builds the configuration space, so its cost is set by how many paths survive, not by
D = d^(2(N-1)). This measures that: for a sequence of cutoffs, the surviving set size, the number of path
weights evaluated, and the RDM, at time steps far past the point where enumerating D is possible.
Accuracy is judged by convergence in the cutoff: the RDM at each cutoff against the smallest cutoff that
still fits, with the brute-force value quoted wherever D is small enough to compute it.
  [Input]  argv: N_max (default 12), cap on paths per slice (default 400000)
  [Output] Data/<System>/OFPF/ofpf_reach.json
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, h_bar
from lib.paths import data_path
from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.CX_mc_readout_N6 import all_weights, normalize
from lib.cx_ofpf_lib import ofpf_filtered

PROPERTY = 'OFPF'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = (sim2001(0.72), tls(), dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2))
CUTOFFS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
D_BRUTE = 2 ** 20                                                 # above this the brute-force sum is skipped


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    rows, path = [], os.path.join(ROOT_DIR(), 'ofpf_reach.json')
    for system in CASES:
        d = system['N_state']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        print(f"\n=== {system['name']}  d={d}", flush=True)
        print(f"{'N':>3} {'D':>12} {'cutoff':>9} {'R':>9} {'n_eval':>10} {'sec':>7} {'rho11':>10} {'|-brute|':>9}",
              flush=True)
        for N in range(2, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            D = d ** (2 * (N - 1))

            brute = None                                           # the reference, where the space still fits
            if D <= D_BRUTE:
                configs = np.array(list(itertools.product(range(d), repeat=2 * (N - 1))), dtype=int)
                Lb = np.array([[all_weights(a, b, configs, U, s, eta).sum() for b in range(d)]
                               for a in range(d)])
                brute = normalize(Lb)
                del configs

            for cutoff in CUTOFFS:
                t0 = time.time()
                try:
                    L, R, n_eval = np.zeros((d, d), complex), [], 0
                    for a in range(d):
                        for b in range(d):
                            cfg, w, ne = ofpf_filtered(a, b, N, U, s, eta, d, cutoff, max_paths=cap)
                            L[a, b] = w.sum()
                            R.append(len(w))
                            n_eval += ne
                except MemoryError as exc:
                    print(f"{N:>3} {D:>12} {cutoff:>9.0e}  skipped: {exc}", flush=True)
                    break
                sec = time.time() - t0
                rho = normalize(L) if abs(np.trace(L)) > 0 else np.full((d, d), np.nan)
                dev = float(np.abs(rho - brute).max()) if brute is not None else float('nan')
                rows.append(dict(system=system['name'], N=N, d=d, D=D, cutoff=cutoff,
                                 R=float(np.mean(R)), n_eval=int(n_eval), sec=sec,
                                 rho=[float(np.real(rho[0, 0])), float(np.real(rho[0, 1])),
                                      float(np.imag(rho[0, 1]))],
                                 dev_brute=dev))
                print(f"{N:>3} {D:>12} {cutoff:>9.0e} {np.mean(R):>9.0f} {n_eval:>10} {sec:>7.1f} "
                      f"{np.real(rho[0, 0]):>10.6f} {dev:>9.1e}", flush=True)
                json.dump(rows, open(path, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

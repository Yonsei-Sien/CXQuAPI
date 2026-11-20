"""The grouping program and the measurement it buys, both as wall-clock time.

The program's cost is already a time: HiGHS reports it.  The measurement it removes becomes a time once a shot
rate is fixed, so this computes the shots a 1% reading of a population needs,
    n = (1 - P_ro) / (eps^2 P_ro),    P_ro = d^{-2(N-1)} |rho_ab|^2 / alpha^2 ,
with rho_ab the unnormalized element from the exact path sum and alpha the rescaling a decomposition not chosen by
the program must carry.  Since the program returns alpha = 1, the difference between alpha > 1 and alpha = 1 is
exactly the measurement time the program removes.  Changing the assumed shot rate scales every shot time by one
common factor and moves no point relative to another.
  [Input]  Data/Grouping/lp_cost_vs_shots.json  (alpha and the program's size and time)
  [Output] Data/Grouping/lp_time_vs_shot_time.json
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, h_bar
from lib.paths import data_path
from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.CX_mc_readout_N6 import all_weights

PROPERTY = 'LP'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
EPS = 0.01                                                  # 1% relative accuracy on a population
SYSTEMS = {'TLS-strong': dict(tls(), sd=ohmic(1.2, 2.5), beta=0.2), 'Sim2001': sim2001(0.72)}
MAX_D = 2_000_000                                           # refuse to enumerate more intermediate configurations


# Unnormalized rho_00 of the exact path sum: the coherent sum of every path's weight
def rho_tilde(system, N):
    d, n_slice = system['N_state'], 2 * (N - 1)
    if d ** n_slice > MAX_D:
        return None
    U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
    s = np.real(np.asarray(system['S_c'], dtype=complex))
    eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
    configs = np.array(list(itertools.product(range(d), repeat=n_slice)))
    return complex(np.sum(all_weights(0, 0, configs, U, s, eta)))


def main():
    rows = json.load(open(os.path.join(ROOT_DIR(), 'lp_cost_vs_shots.json')))
    out = []
    for r in rows:
        sysd = SYSTEMS[r['system']]
        d, N = sysd['N_state'], r['N']
        rt = rho_tilde(sysd, N)
        if rt is None:
            print(f"  {r['system']} N={N}: D too large, skipped", flush=True)
            continue
        P1 = d ** (-2 * (N - 1)) * abs(rt) ** 2                     # readout probability of the exact construction
        rec = dict(system=r['system'], N=N, d=d, rho_tilde=abs(rt), P_ro_exact=P1,
                   lp_seconds=r['exact']['seconds'], lp_ineq=r['exact']['n_ineq'], g=r['exact']['g'],
                   cx=r['exact'].get('cx'))
        for key, alpha in (('componentwise', r['alpha_componentwise']), ('pairs', r['pairs']['alpha']),
                           ('program', 1.0)):
            P = P1 / alpha ** 2
            rec[f'shots_{key}'] = (1 - P) / (EPS ** 2 * P)
        out.append(rec)
        print(f"  {r['system']:<11} N={N}  |rho|={abs(rt):.4f}  P_ro(alpha=1)={P1:.5g}"
              f"  shots: exact {rec['shots_program']:.3g}  pairs {rec['shots_pairs']:.3g}"
              f"  one-per-piece {rec['shots_componentwise']:.3g}", flush=True)
    json.dump(out, open(os.path.join(ROOT_DIR(), 'lp_time_vs_shot_time.json'), 'w'), indent=1)
    print(f"\nwritten; checkpoint: TLS-strong N=3 P_ro should be 0.04132", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

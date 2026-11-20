"""Measurement cost: shots for 1% relative accuracy on a population, n = 1 / (eps^2 P_ro), with
  P_ro = d^{-2(N-1)} rho^2 alpha^{-2}     (CX-QuAPI, this work)
  P_ro = d^{-4(N-1)} rho^2 lambda^{-2}    (dilated propagator of Seneviratne et al.)
rho is the exact population of the classical path sum at that N, alpha and lambda are the measured rescalings.
  [Input]  Data/Common/Cost/fig2_cost.json (full memory), Data/<System>/Cost/cost_memory.json
  [Output] Data/<System>/Shots/shot_cost.json : per method, N: alpha, rho, P_ro, shots
"""
import json
import os
import sys

import numpy as np

from lib.cx_systems_lib import tls, ohmic, run_fpf
from lib.paths import data_path, read_across, read_one

PROPERTY = 'Shots'      # data lands in Data/<System>/Shots/, one file per system
EPS = 0.01
SYSTEMS = {'TLS': tls(), 'TLS-strong': dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2)}


def main():
    cost = read_one('Common', 'Cost', 'fig2_cost.json')                       # full memory, alpha per method
    mem = read_across('Cost', 'cost_memory.json', tuple(SYSTEMS))             # truncated memory, per system
    rho = {(name, N): float(np.real(run_fpf(system, N)[0, 0]))
           for name, system in SYSTEMS.items() for N in range(1, 9)}
    rows = []
    for r in cost:
        system = 'TLS-strong' if 'strong' in r['method'] else 'TLS'
        wang = r['method'].startswith('wang')
        N, p = r['N'], rho[(system, r['N'])]
        P = 2.0 ** (-(4 if wang else 2) * (N - 1)) * p ** 2 / r['alpha'] ** 2
        rows.append(dict(kind='full', method=r['method'], system=system, N=N, K=N, alpha=r['alpha'],
                         rho=p, P_ro=P, shots=1.0 / (EPS ** 2 * P)))
    for r in mem:
        N, p = r['N'], rho[(r['system'], r['N'])]
        P = 2.0 ** (-2 * (N - 1)) * p ** 2 / r['alpha'] ** 2
        rows.append(dict(kind='memory', method=f"K={r['K']}", system=r['system'], N=N, K=r['K'], alpha=r['alpha'],
                         rho=p, P_ro=P, shots=1.0 / (EPS ** 2 * P)))
    # One file per system, beside the rest of that system's numbers
    for tag in sorted(SYSTEMS):
        out = os.path.join(data_path(tag, PROPERTY), 'shot_cost.json')
        json.dump([r for r in rows if r['system'] == tag], open(out, 'w'), indent=1)
        print(f'written to {out}', flush=True)
    for r in rows:
        if r['N'] == 8:
            print(f"{r['system']:11s} {r['method']:16s} alpha={r['alpha']:10.4g} P={r['P_ro']:.3e} "
                  f"shots={r['shots']:.3e}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

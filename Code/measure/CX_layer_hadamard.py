"""Reconstruct the 'layer' column of Table 3 under its stated definition: the Hadamards the per-slice
readout uses against the Hadamards of the full layer.

The full readout prepares uniform_prep(d, n_q) on every slice -- Hadamards on all n_q qubits when d is a
power of two, a transpiled StatePreparation otherwise. A slice whose threshold keeps c of the d states
prepares uniform_prep(c, n_q) instead, and keeps nothing at all when c = 1. This script weighs every slice
by that gate count and sweeps the threshold, under the selection rule of CX_mc_readout_product.py (one
Metropolis seed for every element, no mirror between (a,b) and (b,a)) -- the script whose docstring calls
the layer 'the cost of the Hadamard layer it replaces', and so the likely source of the published column.

  [Input]  argv: Metropolis samples (default 3000; 0 = exact marginals)
  [Output] Data/<System>/Readout/layer_hadamard_ns{n_sample}.json
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, metropolis_configs, uniform_prep, h_bar
from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.CX_mc_readout_N6 import all_weights
from lib.CX_mc_readout_product import marginals
from lib.paths import data_path

PROPERTY = 'Readout'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = ((tls(), 8), (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), 8), (sim2001(0.72), 6))
THRESHOLDS = (0.0, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)

PUB = {'TLS': {2: 0.00, 3: 0.63, 4: 1.00, 5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00},
       'TLS-strong': {2: 0.00, 3: 0.50, 4: 0.58, 5: 0.63, 6: 0.65, 7: 0.67, 8: 0.71},
       'Sim2001(beta=0.72)': {2: 0.44, 3: 0.44, 4: 0.48, 5: 0.45, 6: 0.45}}


# one-qubit gates uniform_prep spends on a uniform state over c of the register's states; c = 1 is a basis
# state, which the readout string carries and no preparation is needed for
def prep_gates(c, n_q):
    if c <= 1:
        return 0
    circ = uniform_prep(c, n_q)
    return sum(n for inst, n in circ.count_ops().items() if inst != 'cx')


N_SAMPLE = 3000        # a driver sets this; 0 means exact marginals, which is what Table 3 quotes


def main():
    n_sample = N_SAMPLE if len(sys.argv) <= 1 else int(sys.argv[1])
    n_qb = {2: 1, 3: 2}
    cost = {d: {c: prep_gates(c, n_qb[d]) for c in range(1, d + 1)} for d in (2, 3)}
    print('one-qubit gates of uniform_prep:', cost, flush=True)
    got = {}
    for system, N_max in CASES:
        d, name = system['N_state'], system['name']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        for N in range(2, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            n_site = 2 * (N - 1)
            configs = np.array(list(itertools.product(range(d), repeat=n_site)), dtype=int)
            index = {tuple(c): i for i, c in enumerate(configs)}
            fr = {}
            for a in range(d):
                for b in range(d):
                    w = all_weights(a, b, configs, U, s, eta)
                    if n_sample:                        # one seed for every element, as the product script does
                        seen = sorted(index[p] for p in metropolis_configs(lambda p: w[index[p]], n_site, d,
                                                                           n_sample, seed=1))
                        m = marginals(w, configs, d, np.array(seen))
                    else:
                        m = marginals(w, configs, d)
                    fr[(a, b)] = m / m.sum(axis=1, keepdims=True)
            for th in THRESHOLDS:
                gates, blocks, tot = 0, 0, 0
                for a in range(d):
                    for b in range(d):
                        f = fr[(a, b)]
                        for k in range(n_site):
                            c = len(set(np.flatnonzero(f[k] >= th)) | {int(np.argmax(f[k]))})
                            gates += cost[d][c]
                            blocks += (c > 1)
                            tot += 1
                got.setdefault(('gates', th), {}).setdefault(name, {})[N] = gates / (tot * cost[d][d])
                got.setdefault(('blocks', th), {}).setdefault(name, {})[N] = blocks / tot

    scored = sorted(((max(abs(p[nm][N] - v) for nm, tab in PUB.items() for N, v in tab.items()), k, p)
                     for k, p in got.items()), key=lambda r: r[0])
    print(f'\nestimator: {"Metropolis seed=1, " + str(n_sample) if n_sample else "exact marginals"}')
    print(f'{"max|err|":>9} {"reading":>8} {"threshold":>10}')
    for err, (defn, th), _ in scored[:6]:
        print(f'{err:>9.3f} {defn:>8s} {th:>10.3f}')
    err, key, best = scored[0]
    print(f'\nbest: {key}')
    for nm, tab in PUB.items():
        print(f'  {nm:>19s} pub  ' + ' '.join(f'{v:5.2f}' for _, v in sorted(tab.items())))
        print(f'  {"":>19s} got  ' + ' '.join(f'{best[nm][N]:5.2f}' for N in sorted(tab)))
    os.makedirs(ROOT_DIR(), exist_ok=True)
    json.dump([dict(err=e, reading=k[0], threshold=k[1],
                    values={nm: {str(N): v for N, v in dd.items()} for nm, dd in p.items()})
               for e, k, p in scored[:12]], open(os.path.join(ROOT_DIR(), f'layer_hadamard_ns{n_sample}.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

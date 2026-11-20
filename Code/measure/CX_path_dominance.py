"""How concentrated is the path sum, and does strong dissipation concentrate it?

The shot saving quoted for the strongly coupled bath is exactly (rho_strong/rho_weak)^2: at the optimal grouping
alpha = 1 for both and the dilution d^{-2(N-1)} is shared, so nothing but the size of the element survives.  That
makes it a statement about how big the answer happens to be, which is close to a tautology for amplitude readout
and moves with the coherent oscillation of the weakly coupled system.

Concentration of the path sum is the claim that was actually meant, and it can be measured without reference to the
size of anything.  Two numbers, both read off the path weights w_i of one element:

  participation ratio  PR = (sum|w|)^2 / sum|w|^2 -- the effective number of paths carrying the sum, cutoff-free
  R(eps)               the fewest top-|w| paths whose coherent sum reproduces the element to a relative eps

A bath that suppresses blips leaves fewer paths with weight, so both fall.  Neither depends on how large the
element is, so a saving read from them is a property of the dynamics rather than of the moment it is read at.

The two named baths differ in all three of (xi, w_c, beta), so comparing them cannot say which one does the work --
the same objection the group-size scan had to answer, since g* turned out to be exactly invariant under xi.  The
second scan here holds the bath shape fixed and moves the coupling alone.  If concentration is the coupling's doing
it must fall with xi at fixed (w_c, beta), and it must do so for either shape.
  [Input]  argv: cap on N for every system (default: the per-system cap in CASES), N for the
           coupling scan (default 6)
  [Output] Data/CouplingSweep/path_dominance.json, path_dominance_xi.json
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, h_bar
from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.CX_mc_readout_N6 import all_weights
from lib.paths import data_path

PROPERTY = 'Bath'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
WEAK   = tls()                                                                 # xi 0.1, w_c 7.5, beta 5.0
STRONG = dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2)          # xi 1.2, w_c 2.5, beta 0.2

# The three systems of Table 3, each paired with the longest propagation that table quotes. The three-state
# model stops at N = 6 because its space is d^{2(N-1)} = 3^{10} there, the same place the readout study stops.
# A driver may override CASES; argv[1] caps every entry, so one argument still shortens the whole scan.
CASES = ((WEAK, 8), (STRONG, 8), (sim2001(0.72), 6))
XI_SCAN = True        # the Common driver leaves this on; a per-system driver turns it off


# Effective number of paths carrying the sum: PR = 1 for a single path, D for D paths of equal weight
def participation(w):
    a = np.abs(w)
    return float(a.sum() ** 2 / (a ** 2).sum())


# Fewest top-|w| paths whose coherent sum is within eps of the whole, relative
def paths_for(w, eps):
    order = np.argsort(-np.abs(w))
    partial = np.cumsum(w[order])
    total = w.sum()
    hit = np.flatnonzero(np.abs(partial - total) <= eps * abs(total))
    return int(hit[0] + 1) if len(hit) else len(w)


# Participation ratio and R(eps) for one system at one propagation length
def measure(system, N):
    d = system['N_state']
    U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
    s = np.real(np.asarray(system['S_c'], dtype=complex))
    eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
    configs = np.array(list(itertools.product(range(d), repeat=2 * (N - 1))))
    w = all_weights(0, 0, configs, U, s, eta)                                  # the element the shot count used
    return len(configs), participation(w), paths_for(w, 1e-2), paths_for(w, 1e-3), float(np.real(w.sum()))


# Coupling alone, at a bath shape held fixed: the control the two named baths cannot provide
def xi_scan(N):
    rows = []
    for shape, w_c, beta in (('fast bath', 7.5, 5.0), ('slow bath', 2.5, 0.2)):
        print(f'{shape}  w_c={w_c}  beta={beta}  N={N}', flush=True)
        for xi in (0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0):
            system = dict(tls(), name=f'{shape} xi={xi}', sd=ohmic(xi, w_c), beta=beta)
            D, pr, r2, r3, rho = measure(system, N)
            rows.append(dict(shape=shape, w_c=w_c, beta=beta, xi=xi, N=N, D=D,
                             participation=pr, R_1pc=r2, R_01pc=r3, rho=rho))
            print(f'  xi={xi:<5g}  PR {pr:7.2f} ({pr / D:6.2%})   R(0.1%) {r3:6d} ({r3 / D:6.2%})'
                  f'   rho11 {rho:7.4f}', flush=True)
    return rows


def main():
    N_cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(ROOT_DIR(), exist_ok=True)   # data_path already does this; kept for symmetry
    rows = []
    for system, N_top in CASES:
        N_max = min(N_top, N_cap) if N_cap else N_top
        d = system['N_state']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        print(f"{system['name']}  (xi, w_c, beta) as configured", flush=True)
        for N in range(2, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            configs = np.array(list(itertools.product(range(d), repeat=2 * (N - 1))))
            D = len(configs)
            w = all_weights(0, 0, configs, U, s, eta)                          # the element the shot count used
            pr, r2, r3 = participation(w), paths_for(w, 1e-2), paths_for(w, 1e-3)
            rows.append(dict(system=system['name'], N=N, D=D, rho=float(np.real(w.sum())),
                             participation=pr, R_1pc=r2, R_01pc=r3))
            print(f'  N={N}  D={D:7d}   PR {pr:9.1f} ({pr / D:7.2%})   '
                  f'R(1%) {r2:7d} ({r2 / D:7.2%})   R(0.1%) {r3:7d} ({r3 / D:7.2%})', flush=True)
    if rows:
        json.dump(rows, open(os.path.join(ROOT_DIR(), 'path_dominance.json'), 'w'), indent=1)

    # Both two-level baths have to be present for the comparison to mean anything, so a driver that
    # names one system prints its own table and skips this.
    both = [N for s, N in CASES if s['name'] in (WEAK['name'], STRONG['name'])]
    if len(both) >= 2:
        print('\nstrong / weak, at equal N')
        N_both = min(both)
        for N in range(2, (min(N_both, N_cap) if N_cap else N_both) + 1):
            a = next(r for r in rows if r['system'] == WEAK['name'] and r['N'] == N)
            b = next(r for r in rows if r['system'] == STRONG['name'] and r['N'] == N)
            print(f'  N={N}   PR {a["participation"]:9.1f} -> {b["participation"]:9.1f}  '
                  f'({a["participation"] / b["participation"]:6.1f}x fewer)   '
                  f'R(1%) {a["R_1pc"]:6d} -> {b["R_1pc"]:6d}  ({a["R_1pc"] / max(b["R_1pc"], 1):6.1f}x)', flush=True)

    # The coupling scan is not a property of either named system -- it holds a bath shape fixed and moves
    # xi -- so it is measured on its own and filed apart from the three named systems.
    if XI_SCAN:
        print('\ncoupling alone, bath shape held fixed')
        N_xi = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        xi_rows = xi_scan(N_xi)
        json.dump(xi_rows, open(os.path.join(data_path('Common', PROPERTY), 'path_dominance_xi.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

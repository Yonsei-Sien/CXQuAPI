"""What the grouping program costs classically, against the measurement cost it removes.

The claim the manuscript makes is that no classical object the construction builds is larger than the circuit it
produces, so the whole accounting is of quantum resources.  This measures both sides of that statement in
commensurable units -- counts of things that must be done.  On the classical side: the number of inequalities and
variables of the linear program, and the wall-clock time HiGHS takes on it.  On the quantum side: alpha^2, the
factor in shots that a decomposition not chosen by the program would have to pay, since the program returns
alpha = 1.
  [Input]  argv: largest N for the two-level system (default 9), largest N for the three-state model (default 7)
  [Output] Data/Grouping/lp_cost_vs_shots.json : per system and N, program size and time at the pairwise family and
           at the smallest exact family, with alpha of both naive cuts
"""
import json
import os
import sys
import time

import numpy as np

from lib.cx_systems_lib import tls, sim2001, ohmic
from measure.CX_coupling_sweep import M_matrix, LAT_TLS, LAT_3LS
from lib.cx_grouping_lp import optimal_grouping, pairs_family, window_family, ucry_cnot
from lib.paths import data_path

PROPERTY = 'LP'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/

# Longest propagation to solve per system. A driver overrides this together with CASES; argv still wins,
# so one argument shortens a run without editing anything.
TOPS = {'TLS-strong': 9, 'Sim2001': 8}


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
MAX_INEQ = 300_000                       # the dense A_ub is n_ineq x n_var floats: refuse to build more than this
CASES = ((dict(tls(), sd=ohmic(1.2, 2.5), beta=0.2), LAT_TLS, 1, 'TLS-strong'),
         (sim2001(0.72), LAT_3LS, 2, 'Sim2001'))


# Size of the program for a family of groups, counted exactly as optimal_grouping lays it out
def program_size(groups, n_lat):
    n_var = sum(len(G) * (len(G) + 1) // 2 for G in groups) + len(groups)      # X_G upper triangles, one slack each
    n_ineq = sum(n_lat ** len(G) for G in groups)                              # one per lattice vector per group
    return n_var, n_ineq


# alpha of the cut that gives every influence coefficient a piece of its own: ln alpha = sum of all the couplings
def alpha_componentwise(M, lattice):
    worst = 2 * max(abs(a * b) for a in lattice for b in lattice)
    n = len(M)
    return float(np.exp(sum(abs(M[k, j]) * worst for k in range(n) for j in range(k + 1, n))))


# Solve once and report alpha, the program size and the wall-clock time, or None if the program is too large to build
def solve(M, groups, lattice):
    n_var, n_ineq = program_size(groups, len(lattice))
    if n_ineq > MAX_INEQ:
        return dict(n_var=n_var, n_ineq=n_ineq, seconds=None, alpha=None, skipped=True)
    t0 = time.perf_counter()
    alpha, _, _ = optimal_grouping(M, groups, lattice)
    return dict(n_var=n_var, n_ineq=n_ineq, seconds=time.perf_counter() - t0,
                alpha=float(alpha), skipped=False)


def main():
    tops = dict(TOPS)
    if len(sys.argv) > 1:
        tops['TLS-strong'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        tops['Sim2001'] = int(sys.argv[2])
    os.makedirs(ROOT_DIR(), exist_ok=True)   # data_path already does this; kept for symmetry
    rows = []
    for system, lattice, n_q, tag in CASES:
        print(f'=== {tag}  (d={system["N_state"]}, |Lambda|={len(lattice)})', flush=True)
        for N in range(2, tops[tag] + 1):
            M = M_matrix(system, N)
            n = N + 1
            rec = dict(system=tag, N=N, n_q=n_q,
                       alpha_componentwise=alpha_componentwise(M, lattice),
                       pairs=solve(M, pairs_family(n), lattice))
            # smallest family reaching alpha = 1, and what that program costs
            rec['exact'] = None
            for g in range(2, n + 1):
                fam = pairs_family(n) if g == 2 else window_family(n, g)
                r = solve(M, fam, lattice)
                if r['skipped'] or r['alpha'] < 1 + 1e-9:
                    # ucry_cnot needs no solve, so the circuit cost is known even where the program is not built
                    rec['exact'] = dict(r, g=g, cx=int(ucry_cnot(fam, n_q)))
                    if not r['skipped']:
                        # re-time the winning family: a single run at the millisecond scale is scipy overhead,
                        # not the program, so take the best of three
                        rec['exact']['seconds'] = min(solve(M, fam, lattice)['seconds'] for _ in range(3))
                    break
            rows.append(rec)
            e, p = rec['exact'], rec['pairs']
            print(f"  N={N:>2}  alpha_cw={rec['alpha_componentwise']:.3g}  alpha_pairs={p['alpha']:.4g}"
                  f"  | pairs: {p['n_ineq']:>7} ineq {p['seconds']:.3f}s"
                  f"  | g*={e['g']}: {e['n_ineq']:>8} ineq "
                  + ('NOT BUILT' if e['skipped'] else f"{e['seconds']:.3f}s") + f"  CNOT={e['cx']:.3g}", flush=True)
    out = os.path.join(ROOT_DIR(), 'lp_cost_vs_shots.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print(f'\nwritten to {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

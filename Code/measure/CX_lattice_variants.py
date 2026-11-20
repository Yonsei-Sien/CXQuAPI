"""Which points of the blip lattice are the ones that force the three-state model to decline to split?

The grouping program is exact because the circuit only ever sees Delta in Lambda^|G|, with Lambda the finite set
of blip values {s_i - s_j}. For the three-state model s = (-1, 0, 1), so Lambda = {-2, -1, 0, 1, 2} and the
program must hold every piece non-negative on all five. That system is also the one where the program declines
to split at all: g* = N + 1 at every step (Data/Sim2001/Grouping/gstar_vs_N.json). Whether that is a property
of M or of the lattice M is tested against is a question this scan answers, by rerunning the same M against
lattices it does not physically have:

    full  {-2,-1,0,1,2}   the physical lattice, the baseline
    ext   {-2, 0, 2}      extremes only, the interior values dropped
    unit  {-1, 0, 1}      ext scaled by one half, so the same set of directions
    pm1   {-1, 1}         the two-point lattice, neither the extremes nor zero

None of the three variants is a physical system. They are diagnostics, and they separate two things that the
physical lattice confounds. A positive rescaling Lambda -> c Lambda cannot change whether alpha = 1 is
reachable, since -Delta^T X_G Delta <= 0 is homogeneous of degree two in Delta; it only sends the exponent to
c^2 times itself, hence alpha -> alpha^(c^2). So unit must return exactly alpha(ext)^(1/4) at every g and the
same g*, and the comparison measures the arithmetic rather than the condition. What pm1 drops instead is the
vectors with a zero component, and those are the ones that impose the same non-negativity on every principal
submatrix of X_G -- a condition on the subgroups of G, not on G. Any change in g* is therefore attributable to
the loss of zero and not to the loss of the extremes.

  [Input]  argv: largest N for the three-state model (default 7)
  [Output] Data/<System>/Grouping/lattice_variants.json : one row per (lattice, N, g) with the alpha the
           program returns there, so that g* is the smallest g on each lattice with alpha = 1
"""
import json
import os
import sys
import time

import numpy as np

from lib.paths import data_path
from lib.cx_systems_lib import sim2001
from lib.cx_grouping_lp import optimal_grouping, pairs_family, window_family, ucry_cnot
from measure.CX_coupling_sweep import M_matrix, LAT_3LS

PROPERTY = 'Grouping'
SYSTEM = 'Common'          # the driver in Running/ sets this

# the physical lattice and the two diagnostics, in the order the table prints them
LATTICES = (('full', LAT_3LS),
            ('ext', np.array([-2.0, 0.0, 2.0])),
            ('unit', np.array([-1.0, 0.0, 1.0])),
            ('pm1', np.array([-1.0, 1.0])))
CASES = ((sim2001(0.72), 2, 'Sim2001'),)
TOL = 1e-9                 # alpha = 1 to within the LP's own accuracy


def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)


# The number of inequality rows the program will carry, summed over groups: one per +-pair of lattice vectors.
# Checked before any family is built, since the count is |Lambda|^|G| and a careless g is an out-of-memory.
# The zero vector is dropped only if the lattice contains zero at all, which pm1 does not.
#   [Input]  groups : list of tuples of time indices;  lattice : the allowed blip values
#   [Output] rows : total inequalities the LP will hold
def program_rows(groups, lattice):
    n_lat = len(lattice)
    zero = 1 if np.any(lattice == 0.0) else 0
    return sum((n_lat ** len(G) - zero) // 2 for G in groups)


# alpha for every group size on one lattice, and the smallest size reaching alpha = 1
#   [Input]  M : (n, n) blip matrix;  lattice : allowed blip values;  n_q, N : qubits per slice, steps
#   [Output] rows : per-g records;  g_star : smallest g with alpha = 1, else N + 1
def scan_one(M, lattice, n_q, N, tag, lat_name, max_rows=4_000_000):
    n, rows, g_star = N + 1, [], None
    for g in range(2, n + 1):
        fam = pairs_family(n) if g == 2 else window_family(n, g)
        n_rows = program_rows(fam, lattice)
        if n_rows > max_rows:
            print(f'    g={g:2d}  skipped: {n_rows} inequalities exceeds the cap', flush=True)
            break
        t0 = time.time()
        alpha, _, _ = optimal_grouping(M, fam, lattice)
        exact = alpha < 1 + TOL
        if exact and g_star is None:
            g_star = g
        rows.append(dict(system=tag, lattice=lat_name, N=N, g=g, slices=n, n_ineq=int(n_rows),
                         cx=int(ucry_cnot(fam, n_q)), alpha=float(alpha), exact=bool(exact),
                         seconds=time.time() - t0))
        print(f'    g={g:2d}  ineq={n_rows:<8d} alpha={alpha:.6g}{"  <- alpha = 1" if exact else ""}'
              f'  [{time.time() - t0:.1f}s]', flush=True)
        if exact:
            break
    return rows, (g_star if g_star is not None else n)


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    os.makedirs(ROOT_DIR(), exist_ok=True)
    rows, summary = [], []
    for system, n_q, tag in CASES:
        for N in range(2, N_max + 1):
            M = M_matrix(system, N)
            for lat_name, lattice in LATTICES:
                print(f'{tag}  N={N}  lattice={lat_name} ({len(lattice)} values)', flush=True)
                got, g_star = scan_one(M, lattice, n_q, N, tag, lat_name)
                rows += got
                a_pair = next((r['alpha'] for r in got if r['g'] == 2), None)
                summary.append(dict(system=tag, lattice=lat_name, N=N, g_star=int(g_star),
                                    alpha_pair=a_pair))
                json.dump(rows, open(os.path.join(ROOT_DIR(), 'lattice_variants.json'), 'w'), indent=1)

    # the comparison the supplementary note reports: g* and alpha(pairs) per lattice, side by side
    print('\n  N  ' + '  '.join(f'{nm:>22s}' for nm, _ in LATTICES), flush=True)
    for N in range(2, N_max + 1):
        cells = []
        for lat_name, _ in LATTICES:
            r = next((s for s in summary if s['N'] == N and s['lattice'] == lat_name), None)
            cells.append('n/a'.rjust(22) if r is None else
                         f"g*={r['g_star']:<2d} a(pairs)={r['alpha_pair']:.4g}".rjust(22))
        print(f'{N:3d}  ' + '  '.join(cells), flush=True)
    # The table above is printed and not filed: the supplementary note quotes it, and the per-g rows it
    # summarizes are already in lattice_variants.json. The summary file it used to write was read by
    # nothing and was removed on 2026-11-18.
    return 0


if __name__ == '__main__':
    sys.exit(main())

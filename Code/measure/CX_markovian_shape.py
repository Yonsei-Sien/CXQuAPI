"""Does the three-state model need larger groups than the two-level one, or does its bath?

Table "Smallest group of two-level system" shows g* falling to 2 once the bath correlation time drops below
one time step, and the three-state model of the main text sits at the other end: g* = N + 1 at every step.
The two facts are usually read together as "a wider state space costs larger groups", but the influence
matrix is built from the bath alone -- gen_M_matrix reads (dt, beta, J) and never touches H or S -- so at a
common dt the two systems hand the program the SAME M and differ only in the lattice the pieces must be
non-negative on:

    two-level    s = (+1, -1)      Lambda = {-2, 0, 2}
    three-state  s = (-1, 0, 1)    Lambda = {-2, -1, 0, 1, 2}

This scan therefore moves one bath across the Markovian boundary and solves the same program on both
lattices at every bath. 1/(w_c dt) is the number of time slices inside one bath correlation time, so the
top of the grid is a bath whose memory dies well inside a single step and the bottom is the bath of the
three-state model itself. What the comparison isolates is the state space: g* can only differ between the
two columns through Lambda, and the CNOT column carries the other half of the dimension's price, since one
group of g slices costs 2^(2 g n_q) with n_q = 1 for the two-level system and 2 for the three-state one.

  [Input]  argv: N (default 5, matching the two-level table), xi (default 1.333, the three-state coupling)
  [Output] Data/<System>/Bath/markovian_shape.json : one row per bath, with g*, alpha(pairs) and the CNOT
           count of the exact grouping on each of the two lattices, plus r and g* at each length in N_R
"""
import json
import os
import sys

import numpy as np

from lib.cx_systems_lib import sim2001, ohmic
from measure.CX_coupling_sweep import M_matrix, LAT_TLS, LAT_3LS
from lib.cx_grouping_lp import optimal_grouping, pairs_family, window_family, ucry_cnot
from lib.paths import data_path

PROPERTY = 'Bath'
SYSTEM = 'Common'          # the driver in Running/ sets this; the scan spans two systems

DT = 0.3                   # common time step, so that both systems see one and the same M
# slices inside one bath correlation time, 1/(w_c dt): the Markovian end first, the model's own bath last
STEPS_PER_TAU = (0.1, 0.25, 0.5, 1.0, 2.0, 1.0 / (1.0 * DT))
BETAS = (5.0, 0.720, 0.2)  # the three-state model's own temperature in the middle
LATTICES = (('TLS', LAT_TLS, 1), ('3LS', LAT_3LS, 2))
# Inequalities the program is allowed to carry in one solve. A window of g slices puts (|Lambda|^g - 1)/2
# of them on its own group, so on the wide lattice the count is 5^g/2. The cap is set from what the node
# holds rather than from the row count alone: the rows enter linprog as a dense array, and measurement puts
# the peak at about eight times its size (89 GB at the 9.8e6 rows of g = 10, N = 10), so 1e7 is the largest
# solve that stays well inside memory. g = N + 1 needs no solve at all; see scan_lattice.
MAX_ROWS = 10_000_000


def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)


# Rows the program carries on the widest lattice, checked before any family is built: |Lambda|^g per group
#   [Input]  n : number of time slices, N + 1;  n_lat : size of the lattice
#   [Output] the largest inequality count any single group size will reach
def worst_case_rows(n, n_lat):
    return max((n_lat ** g) * max(1, n - g + 1) for g in range(2, n + 1))


# r of Eq. (eq:dominance): min_k M_kk / sum_{k' != k} |M_kk'|, the minimum over every slice, boundaries
# included. It is scale free, M -> cM leaving it alone, which is why g* is invariant under the coupling.
# The manuscript reads r at several lengths and g* at N = 5: a short M has fewer off-diagonal entries to
# sum and so reports the bath as more local than it is, while g* needs a length at which the group size
# still has room to move. Reporting r at N = 5, 7 and 10 shows whether the classification by r = 1 is
# stable in the length it is read at. g* follows to N = 10 as well: the group sizes below N + 1 are solved
# outright up to MAX_ROWS, and g = N + 1 itself is settled by the argument in scan_lattice, so no cell of
# the table is left as a bound.
#   [Input]  M : (n, n) blip matrix
#   [Output] r : the diagonal dominance, r >= 1 meaning M is diagonally dominant
def dominance(M):
    off = np.abs(M).sum(axis=1) - np.abs(np.diag(M))
    return float(np.min(np.diag(M) / off))


# C(M): the diagonal of M with the off-diagonals replaced by -|M_kk'|. M is an H-matrix, that is scaled
# diagonally dominant, iff C(M) >= 0, which is the scale-free form of the dominance r tests plainly.
#   [Output] the least eigenvalue of C(M); its SIGN is what the comparison in the text rests on
def comparison_lmin(M):
    C = -np.abs(M)
    np.fill_diagonal(C, np.diag(M))
    return float(np.linalg.eigvalsh(C).min())


# Rows one family puts on the program: each group of size |G| carries (|Lambda|^|G| - 1)/2 inequalities,
# one per lattice vector with the +-pairs identified and the zero vector dropped.
#   [Input]  fam : list of groups;  n_lat : size of the lattice
#   [Output] the inequality count, which is checked BEFORE the family is built
def family_rows(fam, n_lat):
    return sum((n_lat ** len(G) - 1) // 2 for G in fam)


# alpha at every group size up to the first one reaching alpha = 1, on one lattice. The whole curve is
# kept rather than its endpoint, because the group sizes below g* are where a wider lattice shows itself.
# The scan stops early if the next group size would exceed MAX_ROWS, and then reports the largest size
# PROVEN insufficient instead of g*, so a bounded cell is still a statement about the bath.
#   [Input]  M : blip matrix;  lattice : allowed blip values;  n_q : path qubits per slice;  N : steps;
#            cap : largest inequality count allowed in one solve, None for no cap
#   [Output] {'g': smallest g with alpha = 1 or None if the cap was hit, 'gt': largest g with alpha > 1,
#             'cx': CNOTs of the exact family, 'curve': {g: alpha} for every g solved}
def scan_lattice(M, lattice, n_q, N, cap=None):
    n, curve = N + 1, {}
    for g in range(2, n + 1):
        fam = pairs_family(n) if g == 2 else window_family(n, g)
        # g = n is one group over every slice, so the equality sum_G embed(X_G) = M forces X_G = M and the
        # only question is the sign of -Delta^T M Delta. M is positive definite at every bath of this grid,
        # lambda_min from 1.1e-4 to 1.5 as CX_markovian_full_window measures, so -Delta^T M Delta < 0 for
        # every real Delta, let alone on the lattice. So t_G = 0 and alpha = 1 identically: the single full
        # window is always exact, and solving a 5^n-row program to rediscover it is wasted (it is also where
        # linprog runs out of memory at n = 11). The LP confirms it wherever it is affordable: every cell
        # whose g* reaches n solves to alpha = 1.
        if g == n:
            curve[g] = 1.0
            return dict(g=n, gt=n - 1, cx=int(ucry_cnot(fam, n_q)), curve=curve)
        if cap is not None and family_rows(fam, len(lattice)) > cap:
            return dict(g=None, gt=g - 1, cx=None, curve=curve)
        alpha, _, _ = optimal_grouping(M, fam, lattice)
        curve[g] = float(alpha)
        if alpha < 1 + 1e-9:
            return dict(g=g, gt=g - 1, cx=int(ucry_cnot(fam, n_q)), curve=curve)
    return dict(g=n, gt=n - 1, cx=int(ucry_cnot(window_family(n, n), n_q)), curve=curve)


# g* of one cell as the table prints it: the number, or '>gt' where the cap refused the next group size
def gstr(row, tag, n_r):
    g = row[f'g_{tag}_at'][str(n_r)]
    return str(g) if g is not None else f">{row[f'gt_{tag}_at'][str(n_r)]}"


# [Input] N, xi : taken from argv when run as a script, passed explicitly by a driver that also runs
#                 another scan, since the two read different positional arguments
def main(N=None, xi=None, N_R=(5, 7, 10)):
    N = int(sys.argv[1]) if N is None and len(sys.argv) > 1 else (5 if N is None else N)
    xi = 2 * 2.094 / np.pi if xi is None else xi
    n = N + 1
    print(f'N = {N} ({n} slices), dt = {DT}, xi = {xi:.4g}; '
          f'worst inequality count {worst_case_rows(n, len(LAT_3LS))}', flush=True)

    rows = []
    for beta in BETAS:
        for s_tau in STEPS_PER_TAU:
            w_c = 1.0 / (s_tau * DT)
            # H and S never enter M; one system dict carries the bath for both lattices
            bath = dict(sim2001(beta), dt=DT, beta=beta, sd=ohmic(xi, w_c))
            M = M_matrix(bath, N)
            row = dict(N=N, dt=DT, xi=xi, w_c=float(w_c), beta=beta, steps_per_tau=float(s_tau),
                       beta_w_c=float(beta * w_c), r=dominance(M),
                       r_at={}, N_r=list(N_R), c_lmin=comparison_lmin(M))
            # one M per propagation length: r is read at all of N_R, and so is g* as far as the cap allows
            M_r = {n_r: (M if n_r == N else M_matrix(bath, n_r)) for n_r in N_R}
            row['r_at'] = {str(n_r): dominance(M_r[n_r]) for n_r in N_R}
            # the comparison matrix at each length too: the identity "pairs suffice iff C(M) >= 0" is then
            # testable at every cell of the table, not only at the length g* was first read at
            row['c_lmin_at'] = {str(n_r): comparison_lmin(M_r[n_r]) for n_r in N_R}
            row['r_long'] = row['r_at'].get('7')      # the single length the earlier table was read at
            for tag, lattice, n_q in LATTICES:
                sc = scan_lattice(M, lattice, n_q, N)
                row[f'g_{tag}'] = int(sc['g'])
                row[f'cx_{tag}'] = int(sc['cx'])
                row[f'alpha_pair_{tag}'] = sc['curve'][2]
                row[f'alpha_{tag}'] = {str(k): v for k, v in sc['curve'].items()}
                # g* at every length, with the cap: 'g' is None where the program was refused, and then
                # 'gt' is the largest group size shown to leave alpha > 1, so g* > gt
                row[f'g_{tag}_at'], row[f'gt_{tag}_at'] = {}, {}
                for n_r in N_R:
                    sr = sc if n_r == N else scan_lattice(M_r[n_r], lattice, n_q, n_r, cap=MAX_ROWS)
                    row[f'g_{tag}_at'][str(n_r)] = sr['g']
                    row[f'gt_{tag}_at'][str(n_r)] = sr['gt']
                    row[f'alpha_{tag}_at_{n_r}'] = {str(k): v for k, v in sr['curve'].items()}
            rows.append(row)
            print(f"  beta={beta:>5.3f}  w_c={w_c:>7.3f}  1/(w_c dt)={s_tau:>5.3f}  "
                  f"r=" + " ".join(f"{n}:{row['r_at'][str(n)]:.4f}" for n in N_R) + "  "
                  f"g*(TLS)=" + ",".join(gstr(row, 'TLS', n) for n in N_R) + "  "
                  f"g*(3LS)=" + ",".join(gstr(row, '3LS', n) for n in N_R) + "  "
                  f"alpha(pairs) {row['alpha_pair_TLS']:.5g} / {row['alpha_pair_3LS']:.5g}  "
                  f"CNOT {row['cx_TLS']} / {row['cx_3LS']}", flush=True)
            json.dump(rows, open(os.path.join(ROOT_DIR(), 'markovian_shape.json'), 'w'), indent=1)

    # the claim the table is here to settle: the lattice is the only difference, so look at it directly
    dg = max(abs(r['g_TLS'] - r['g_3LS']) for r in rows)
    da = max(abs(r['alpha_pair_TLS'] - r['alpha_pair_3LS']) /
             max(r['alpha_pair_TLS'], 1e-300) for r in rows)
    print(f'\nlargest difference in g* between the two lattices : {dg}', flush=True)
    print(f'largest relative difference in alpha(pairs)        : {da:.3g}', flush=True)
    print(f'CNOT ratio of the two state spaces at equal g*     : '
          f'{set(r["cx_3LS"] // r["cx_TLS"] for r in rows if r["g_TLS"] == r["g_3LS"])}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

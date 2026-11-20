"""Optimal influence-functional grouping as a linear program on the blip lattice.

The circuit never sees a continuous blip vector: on a basis state x of the path register, the group generator is
Re d_G(x) = -(1/hbar) Delta(x)^T X_G Delta(x), with each Delta_k drawn from the finite set {s_i - s_j}. So the
block-encoding rescaling of group G is exactly

    lambda_G = exp( max(0, max_{Delta in lattice^|G|} -Delta^T X_G Delta) / hbar ),

and lambda_G = 1 iff X_G is nonnegative on the lattice -- WEAKER than X_G >= 0 (which is only sufficient).
Every constraint "t_G >= -Delta^T X_G Delta" is linear in the entries of X_G, so for a fixed family of groups

    min  sum_G t_G     s.t.  sum_G embed(X_G) = M,   t_G >= -Delta^T X_G Delta  for all lattice Delta,   t_G >= 0

is an LP whose optimum gives the smallest subnormalization alpha = exp(sum_G t_G / hbar) reachable by ANY split
of the exponent over those groups (diagonals and couplings alike may be shared between overlapping groups).

  [Input]  M        : (n, n) blip matrix, Re d_tot = -(1/hbar) Delta^T M Delta
           groups   : list of tuples of time indices
           lattice  : 1-D array of allowed blip values (e.g. [-2, 0, 2] for S = sigma_z)
  [Output] alpha, {G: X_G}, {G: t_G}
"""
import itertools

import numpy as np
from scipy.optimize import linprog

from pyqake.ansatz.CXQUAPI import h_bar


def lattice_vectors(lattice, g):
    # one representative of each +-pair (Delta and -Delta give the same quadratic form), zero vector dropped
    seen, out = set(), []
    for v in itertools.product(lattice, repeat=g):
        v = np.asarray(v, dtype=float)
        if not v.any():
            continue
        key = tuple(v); neg = tuple(-v)
        if neg in seen:
            continue
        seen.add(key); out.append(v)
    return out


def optimal_grouping(M, groups, lattice):
    n = len(M)
    # variable layout: for each group, its upper-triangle entries X_G[a, b] (a <= b), then one t_G per group
    index, n_var = [], 0
    for G in groups:
        pairs = [(a, b) for a in range(len(G)) for b in range(a, len(G))]
        index.append({p: n_var + i for i, p in enumerate(pairs)})
        n_var += len(pairs)
    t_index = [n_var + i for i in range(len(groups))]
    n_var  += len(groups)

    # equality: sum_G embed(X_G) = M, entry by entry (upper triangle; off-diagonals appear twice in X, once here)
    eq_rows, eq_rhs = [], []
    for i in range(n):
        for j in range(i, n):
            row = np.zeros(n_var)
            for G, idx in zip(groups, index):
                if i in G and j in G:
                    a, b = sorted((G.index(i), G.index(j)))
                    row[idx[(a, b)]] = 1.0
            if row.any():
                eq_rows.append(row); eq_rhs.append(M[i, j])
            elif abs(M[i, j]) > 1e-14:
                raise ValueError(f'entry ({i},{j}) of M is covered by no group')

    # inequality: -Delta^T X_G Delta - t_G <= 0 for every lattice vector (A_ub x <= b_ub)
    ub_rows = []
    for G, idx, t in zip(groups, index, t_index):
        for v in lattice_vectors(lattice, len(G)):
            row = np.zeros(n_var)
            for (a, b), col in idx.items():
                row[col] = -(v[a] * v[a] if a == b else 2.0 * v[a] * v[b])
            row[t] = -1.0
            ub_rows.append(row)

    cost = np.zeros(n_var); cost[t_index] = 1.0
    bounds = [(None, None)] * (n_var - len(groups)) + [(0.0, None)] * len(groups)
    # a group can only carry a coupling that exists as an IF term: entries where M vanishes (memory truncation)
    # are pinned to zero, so no group needs a Pauli term the influence functional does not have
    for G, idx in zip(groups, index):
        for (a, b), col in idx.items():
            if a != b and abs(M[G[a], G[b]]) < 1e-300:
                bounds[col] = (0.0, 0.0)
    res = linprog(cost, A_ub=np.array(ub_rows), b_ub=np.zeros(len(ub_rows)),
                  A_eq=np.array(eq_rows), b_eq=np.array(eq_rhs), bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(res.message)

    X, T = {}, {}
    for G, idx, t in zip(groups, index, t_index):
        XG = np.zeros((len(G), len(G)))
        for (a, b), col in idx.items():
            XG[a, b] = XG[b, a] = res.x[col]
        X[G], T[G] = XG, res.x[t]
    return float(np.exp(sum(T.values()) / h_bar)), X, T


# -------- group families --------------------------------------------------------------------------------------
def pairs_family(n):
    return [(kp, k) for k in range(n) for kp in range(k)]


# g consecutive times per window, plus every coupling the windows miss as its own pair
def window_family(n, g):
    if g >= n:
        return [tuple(range(n))]
    windows = [tuple(range(i, i + g)) for i in range(n - g + 1)]
    far = [(kp, k) for k in range(n) for kp in range(k) if k - kp >= g]
    return windows + far


def ucry_cnot(groups, n_q):
    # one ancilla + one uniformly controlled Ry per group, on its 2 * |G| * n_q path qubits
    return sum(2 ** (2 * len(G) * n_q) for G in groups)

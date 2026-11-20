"""Populations under the full readout from a superposed initial state: panels c and d of the results figure.

The path sum is linear in the initial reduced density matrix. Writing the forward and backward chains from
their own first slice,

    L_ab = sum_{s0f, s0b} rho0[s0f, s0b] * sum_p  A_f(p) A_b(p)* exp(-Phi(p)/hbar),

so an initial superposition costs one pass of the existing path sum per initial slice pair (s0f, s0b) and
nothing more: no new circuit, no new grouping. For rho(0) = |psi><psi| with psi supported on the set `ends`,
that is |ends|^2 passes, each weighted 1/|ends|.

Only the diagonal of L is needed here, because the panels show populations alone, so the work is d passes
rather than d^2 and the trace of that diagonal normalizes the result.

Why the initial state is a superposition and not a single state:
  - Sim2001 with its published tunnelling V = 0.066667 keeps the middle state empty to 2e-3 forever, since
    V is small against the level spacings 20V and 40V. Raising V fills it. Raising V changes nothing about
    the grouping: the blip matrix M depends on the bath, dt and the position operator S, not on H, so g* and
    alpha are the same at V = 1.0 as at V = 0.066667 (measured: g* = 6, alpha(pairs) = 36.1387 at N = 5 for both).
  - chain(8) is symmetric under site reversal and the influence functional is invariant under s -> -s, so an
    initial state symmetric under 0 <-> 7 forces the eight populations into four exactly degenerate pairs.
    The panel therefore starts from |0> and |4>, which share no such symmetry, and all eight curves separate.

  [Input]  CASES : ((system, N_max, ends), ...) set by the driver in Running/
  [Output] Data/<System>/Readout/super_pops.json : per N, t, the d populations, and the deviation from the
           independent classical path sum QUAPI.FPF wherever that cross-check is affordable
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, h_bar
from pyqake.ansatz.QUAPI import FPF
from lib.paths import data_path

PROPERTY = 'Readout'   # beside results_readout.json, which carries the other panels of the same figure
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/
CASES = ()             # the driver sets this too

# FPF enumerates d^{2(N+1)} paths against the d^{2(N-1)} configurations summed here, so the cross-check is
# dropped once it costs more than the measurement it checks, exactly as CX_results_readout_data does.
FPF_MAX_PATHS = 2e7

# Configurations are summed in blocks of this many rows, so that peak memory is set here and not by d^{2(N-1)}
CHUNK = 2_000_000


def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)


# Rows of the d-ary odometer on n digits, without materializing a list of tuples
#   [Input]  d, n : alphabet size, number of intermediate slices (2(N-1))
#   [Output] (n_rows, n) array of digits, taken a chunk at a time by the caller
def config_chunk(d, n, lo, hi):
    if n == 0:
        return np.zeros((1, 0), dtype=np.int64)
    idx = np.arange(lo, hi, dtype=np.int64)
    return (idx[:, None] // d ** np.arange(n - 1, -1, -1)) % d


# One term of the path sum: the closed-system amplitudes of both branches times the influence functional,
# with the initial slice pinned to (s0f, s0b) and the final slice to (a, b).
#   A_f A_b* = prod_k U[s_{k+1}^+, s_k^+] U[s_{k+1}^-, s_k^-]*
#   Phi      = sum_k (s_k^+ - s_k^-) ( sum_{k' <= k} eta_kk' s_k'^+ - eta_kk'* s_k'^- )
#   [Output] (n_rows,) complex array, one weight per configuration
def weights(a, b, s0f, s0b, cfg, U, s, eta):
    n = len(cfg)
    fwd = np.column_stack([np.full(n, s0f), cfg[:, 0::2], np.full(n, a)])
    bwd = np.column_stack([np.full(n, s0b), cfg[:, 1::2], np.full(n, b)])
    amp = np.prod(U[fwd[:, 1:], fwd[:, :-1]] * np.conj(U[bwd[:, 1:], bwd[:, :-1]]), axis=1)
    s_f, s_b = s[fwd], s[bwd]
    eta_low = np.tril(eta)                                          # k_after >= k_before
    exponent = np.sum((s_f - s_b) * (s_f @ eta_low.T - s_b @ np.conj(eta_low).T), axis=1)
    return amp * np.exp(-exponent / h_bar)


# Populations at N steps from rho(0) = |psi><psi|, psi = sum_{e in ends} |e> / sqrt(|ends|)
#   [Output] (d,) real array summing to one
def pops_super(system, N, ends):
    d = len(system['H_c'])
    U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)   # U = exp(-i H dt / hbar)
    s = np.real(np.asarray(system['S_c'], dtype=complex))              # position eigenvalues, one per state
    eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
    n_site = 2 * (N - 1)                                               # N = 1: one configuration, no sites
    total, w0 = d ** n_site, 1.0 / len(ends)
    diag = np.zeros(d, dtype=complex)
    for lo in range(0, total, CHUNK):
        cfg = config_chunk(d, n_site, lo, min(lo + CHUNK, total))
        for a in range(d):
            for s0f in ends:
                for s0b in ends:
                    diag[a] += w0 * weights(a, a, s0f, s0b, cfg, U, s, eta).sum()
    return np.real(diag / diag.sum())


# Independent cross-check: QUAPI.FPF is a separate classical path sum and takes an arbitrary initial RDM
def fpf_super(system, N, ends):
    d, sd = len(system['H_c']), system['sd']

    class _FPF(FPF):
        def spectral_density(self, w):
            return sd(w)

    psi = np.zeros(d, dtype=complex)
    psi[list(ends)] = 1.0 / np.sqrt(len(ends))
    with tempfile.TemporaryDirectory() as chk, contextlib.redirect_stdout(io.StringIO()):
        f = _FPF(np.outer(psi, psi.conj()), system['H_c'], system['S_c'],
                 N * system['dt'] + 1e-9, system['dt'], N_memory=N, beta=system['beta'])
        f.chkfile = chk
        f.run()
        return np.real(np.diag(np.load(f'{chk}/{np.around(N * system["dt"], 3)}.npy')))


def main():
    rows, path = [], os.path.join(ROOT_DIR(), 'super_pops.json')
    for system, N_max, ends in CASES:
        d = len(system['H_c'])
        for N in range(1, N_max + 1):
            p = pops_super(system, N, ends)
            dev = (float(np.abs(p - fpf_super(system, N, ends)).max())
                   if d ** (2 * (N + 1)) <= FPF_MAX_PATHS else None)
            rows.append(dict(system=system['name'], N=N, t=N * system['dt'], d=d,
                             ends=list(ends), D=d ** (2 * (N - 1)),
                             pops=[float(x) for x in p], dev_fpf=dev))
            print(rows[-1], flush=True)
            json.dump(rows, open(path, 'w'), indent=1)
    print(f'written to {path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Path weights of algorithm 1, read off the amplitude identity instead of off the circuit.

The register amplitude of a configuration p is c_p = w_ab(p) / alpha, so the readout onto the uniform
state over a set S is exact without running the circuit:

    L_S = sum_{p in S} w_ab(p),    P_S = |L_S|^2 / (alpha^2 |S|),    D = d^(2(N-1)).

Every readout measurement under measure/ sums weights with `all_weights` and trace-normalizes with
`normalize`, so both live here and are defined once.

The Monte-Carlo selection sweep this module used to carry (a Metropolis walk on |w| against the R
largest |w|, written to Data/TP/mc_readout_N{N}.json) was removed on 2026-11-18. No notebook and no
registry row read that file; the published readout numbers come from measure/CX_results_readout_data.py.

  [Input]  a, b    : the RDM element being read
           configs : (n_cfg, 2(N-1)) intermediate slice configurations
           U, s    : closed-system propagator and the position eigenvalues
           eta     : influence coefficients eta_kk'
  [Output] all_weights -> (n_cfg,) complex weights; normalize -> trace-normalized RDM
"""
import numpy as np

from pyqake.ansatz.CXQUAPI import h_bar


# All path weights w_ab(p) for p = (f_1, b_1, ..., f_{N-1}, b_{N-1}), vectorized over p
def all_weights(a, b, configs, U, s, eta):
    n_cfg = len(configs)
    fwd = np.column_stack([np.zeros(n_cfg, int), configs[:, 0::2], np.full(n_cfg, a)])
    bwd = np.column_stack([np.zeros(n_cfg, int), configs[:, 1::2], np.full(n_cfg, b)])
    amp = np.prod(U[fwd[:, 1:], fwd[:, :-1]] * np.conj(U[bwd[:, 1:], bwd[:, :-1]]), axis=1)
    s_f, s_b = s[fwd], s[bwd]
    eta_low = np.tril(eta)                                         # k_aft >= k_bef
    exponent = np.sum((s_f - s_b) * (s_f @ eta_low.T - s_b @ np.conj(eta_low).T), axis=1)
    return amp * np.exp(-exponent / h_bar)


# Trace-normalized RDM from per-element sums L_ab
def normalize(L):
    return L / np.trace(L)

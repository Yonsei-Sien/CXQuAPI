"""On-the-fly path filtering (OFPF) of Sim & Makri, in the form the CX-QuAPI readout needs.

The influence functional of QUAPI is causal: in
    w_ab(p) = prod_k U[f_k, f_{k-1}] conj(U[b_k, b_{k-1}])
              * exp(- sum_k (s_{f_k} - s_{b_k}) sum_{k' <= k} (eta_{kk'} s_{f_k'} - conj(eta_{kk'}) s_{b_k'}) / hbar)
the factor carried by slice k involves only slices k' <= k (this is the np.tril(eta) of all_weights).
A prefix of the path therefore has a well-defined weight that later extensions only multiply, never
revise, so the path sum can be grown slice by slice and pruned as it grows -- which is what
QUAPI.FPF does with its `cutoff` and what Sim & Makri call filtering.

Two differences from QUAPI.FPF, both because this builds a readout basis rather than a trajectory:
  - the endpoints (a, b) at slice N are fixed, so only the N-1 intermediate slices are pruned;
  - eta is the one belonging to the target final time N, since a single final RDM is wanted.
With cutoff = 0 the surviving set is the complete configuration space and the weights returned are
exactly those of all_weights -- the test that the incremental build is not an approximation.

  [Output] configs : surviving intermediate configurations, ordered (f_1, b_1, ..., f_{N-1}, b_{N-1})
                     to match CX_results_readout_data.py
           weights : the complete path weights w_ab(p) of those configurations
           n_eval  : path weights evaluated along the way -- the classical cost of the selection
"""
import numpy as np

from pyqake.ansatz.CXQUAPI import h_bar


# One slice of the filtered path sum: extend every surviving prefix by all d^2 branch pairs,
# multiply in the slice-k propagator and influence factors, then drop what falls below the cutoff.
#   [Input]  fwd, bwd : prefixes (n_prefix, k) of forward and backward branch labels, slices 0..k-1
#            w        : prefix weights
#            k        : slice being added
#            cutoff   : absolute threshold on |prefix weight|; 0 keeps everything
#   [Output] the extended and pruned (fwd, bwd, w), plus the number of weights evaluated
def extend_slice(fwd, bwd, w, k, U, s, eta, d, cutoff):
    n = len(w)
    rep = np.repeat(np.arange(n), d * d)                               # which prefix each branch grew from
    f_k = np.tile(np.repeat(np.arange(d), d), n)                       # (f_k, b_k) over the d^2 branch pairs
    b_k = np.tile(np.tile(np.arange(d), d), n)
    F = np.column_stack([fwd[rep], f_k])
    B = np.column_stack([bwd[rep], b_k])

    # U = exp(-i H dt / hbar) on the forward branch, its conjugate on the backward branch
    amp = U[f_k, fwd[rep, -1]] * np.conj(U[b_k, bwd[rep, -1]])

    # influence factor of slice k: the blip (s_f - s_b)_k against all of its own past
    e = eta[k, :k + 1]
    coef = (s[f_k] - s[b_k]) * (s[F] @ e - s[B] @ np.conj(e))
    w_new = w[rep] * amp * np.exp(-coef / h_bar)

    keep = np.ones(len(w_new), bool) if cutoff <= 0 else np.abs(w_new) > cutoff
    return F[keep], B[keep], w_new[keep], len(w_new)


# The filtered path sum for one RDM element, grown from the initial state and pruned at every slice.
#   [Input]  a, b   : final forward and backward labels (the RDM element being read)
#            N      : number of time steps; the path has N-1 intermediate slices
#            cutoff    : absolute threshold on |prefix weight| during the growth
#            max_paths : raise rather than expand past this many paths in one slice
#   [Output] configs, weights, n_eval  (see the module docstring)
def ofpf_filtered(a, b, N, U, s, eta, d, cutoff=0.0, max_paths=None):
    fwd = np.zeros((1, 1), int)                                        # f_0 = b_0 = 0: the initial state
    bwd = np.zeros((1, 1), int)
    w = np.ones(1, complex)
    n_eval = 1

    for k in range(1, N):                                              # intermediate slices only
        if max_paths is not None and len(w) * d * d > max_paths:       # size is known before the expansion
            raise MemoryError(f'slice {k} would hold {len(w) * d * d} paths')
        fwd, bwd, w, ne = extend_slice(fwd, bwd, w, k, U, s, eta, d, cutoff)
        n_eval += ne
        if len(w) == 0:                                                # cutoff swallowed the whole sum
            return np.zeros((0, 2 * (N - 1)), int), np.zeros(0, complex), n_eval

    # slice N carries the fixed endpoints, so it is multiplied in but never pruned
    fwd, bwd, w, ne = extend_slice(fwd, bwd, w, N, U, s, eta, d, 0.0)
    n_eval += ne
    sel = (fwd[:, -1] == a) & (bwd[:, -1] == b)
    fwd, bwd, w = fwd[sel], bwd[sel], w[sel]

    configs = np.empty((len(w), 2 * (N - 1)), int)                     # interleave to (f_1, b_1, f_2, b_2, ...)
    configs[:, 0::2] = fwd[:, 1:-1]
    configs[:, 1::2] = bwd[:, 1:-1]
    return configs, w, n_eval


# Cutoff that makes the filtered set as close as possible to a target size, by bisection on log cutoff.
# Needed only to compare the filter against a sampled selection at equal readout cost.
#   [Input]  target : wanted number of surviving configurations
#   [Output] cutoff, configs, weights, n_eval of the chosen cutoff
def ofpf_at_size(a, b, N, U, s, eta, d, target, n_bisect=40):
    lo, hi = 0.0, 1.0                                                  # lo keeps everything; raise hi until it prunes enough
    for _ in range(200):
        cfg, w, ne = ofpf_filtered(a, b, N, U, s, eta, d, hi)
        if len(w) <= target:
            break
        hi *= 4.0
    best = (hi, cfg, w, ne)
    for _ in range(n_bisect):
        mid = np.sqrt(max(lo, hi * 1e-18) * hi) if lo == 0.0 else np.sqrt(lo * hi)
        cfg, w, ne = ofpf_filtered(a, b, N, U, s, eta, d, mid)
        if len(w) > target:
            lo = mid
        else:
            hi, best = mid, (mid, cfg, w, ne)
    return best


# One cutoff shared by every RDM element, chosen so that the elements together hold a target number of
# paths. Matching element by element instead lets one element's selection starve another: the walk's sets
# can run from 1 path to 1296 across the nine elements of a d = 3 system, and forcing the filter onto each
# of those targets separately drives most elements into a regime the single threshold never visits.
#   [Input]  target_total : wanted sum of the per-element set sizes
#   [Output] cutoff, {(a,b): (configs, weights)}, total paths, weights evaluated
def ofpf_at_total(N, U, s, eta, d, target_total, n_bisect=40):
    def sets_at(cutoff):
        out, tot, ev = {}, 0, 0
        for a in range(d):
            for b in range(d):
                cfg, w, ne = ofpf_filtered(a, b, N, U, s, eta, d, cutoff)
                out[(a, b)] = (cfg, w)
                tot += len(cfg)
                ev += ne
        return out, tot, ev

    lo, hi = 0.0, 1.0                                                  # raise hi until the sum fits
    for _ in range(200):
        sets, tot, ev = sets_at(hi)
        if tot <= target_total:
            break
        hi *= 4.0
    best = (hi, sets, tot, ev)
    for _ in range(n_bisect):
        mid = hi * 1e-9 if lo == 0.0 else np.sqrt(lo * hi)
        sets, tot, ev = sets_at(mid)
        if tot > target_total:
            lo = mid
        else:
            hi, best = mid, (mid, sets, tot, ev)
    return best

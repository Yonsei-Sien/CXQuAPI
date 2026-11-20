"""Data for the results figure: the reduced density matrix read in full, and read in the Monte-Carlo-selected
basis, against the exact classical path sum.

Per system and time step N (full memory throughout; no memory truncation anywhere):
  exact      : rho_11, Re rho_12, Im rho_12 of the full-memory path sum, cross-checked against QUAPI.FPF
  selected   : the same three quantities when only the R paths that survive the threshold zeta are read
  ofpf       : the same three quantities under on-the-fly path filtering, the filter of Sim & Makri grown
               slice by slice and pruned as it grows (cx_ofpf_lib), at one cutoff fixed for every system and
               every step -- the filter's own parameter, in the absolute-threshold convention QUAPI.FPF uses,
               so that the number of surviving paths is an output and not a target
  ofpf_box   : the same when the filter's set is reduced per slice to the states clearing zeta, which is a
               product over slices and so is what one preparation block per slice actually prepares
  ofpf_rank  : the same at the R largest weights -- the filter's own ideal, but a ranking that has to score
               all D paths first, which is the cost the filter exists to avoid
  ofpf_eval  : path weights the filter evaluated, against the D a ranking needs: the classical selection cost
  R, D       : paths kept (mean over the d^2 elements) and paths in total
Both are evaluated from the identity amplitude = classical path weight (verified on circuits to 7e-15).
  [Input]  argv: threshold zeta (default 0.1), Metropolis samples (default 3000), OFPF cutoff (default 1e-6)
  [Output] Data/<System>/Readout/results_readout.json
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, metropolis_configs, h_bar
from lib.cx_systems_lib import tls, sim2001, ohmic, run_fpf
from lib.CX_mc_readout_N6 import all_weights, normalize
from lib.CX_mc_readout_product import marginals
from lib.cx_ofpf_lib import ofpf_filtered
from lib.paths import data_path

PROPERTY = 'Readout'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = ((tls(), 8), (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), 8), (sim2001(0.72), 6))

# One absolute threshold on |prefix weight|, held fixed across the three systems and every step. Chosen so
# that no element moves by more than 2e-4 anywhere while the surviving fraction still runs from 91% of the
# weakly damped sum down to 0.6% of the three-state one: the separation is then the bath's doing and not a
# parameter set per case (Data/<System>/Readout/fixed_cutoff_sweep.json, CX_fixed_cutoff_sweep.py).
CUTOFF = 1e-6

# The independent classical cross-check QUAPI.FPF performs enumerates d^{2(N+1)} paths, which is a
# steeper count than the d^{2(N-1)} configurations this module itself sums: the FPF path carries both
# end points and both branches, the configuration space only the intermediate slices. The check is
# therefore dropped once it costs more than the measurement it checks, and dev_fpf is left None there.
# Every row of the two- and three-state systems stays below this bound, so none of them is affected.
FPF_MAX_PATHS = 2e7


# Mirror of a per-slice keep list under the exchange of the forward and backward branches
def mirror(kept):
    return [kept[k + 1] if k % 2 == 0 else kept[k - 1] for k in range(len(kept))]


# Same exchange applied to whole configurations rather than to per-slice keep lists
def mirror_configs(configs):
    out = configs.copy()
    out[:, 0::2], out[:, 1::2] = configs[:, 1::2], configs[:, 0::2]
    return out


def main():
    zeta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    cut = float(sys.argv[3]) if len(sys.argv) > 3 else CUTOFF
    rows, path = [], os.path.join(ROOT_DIR(), 'results_readout.json')
    for system, N_max in CASES:
        d = system['N_state']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        for N in range(1, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            n_site = 2 * (N - 1)                                            # N=1: one configuration, no sites
            configs = (np.zeros((1, 0), int) if n_site == 0 else
                       np.array(list(itertools.product(range(d), repeat=n_site)), dtype=int))
            index = {tuple(c): i for i, c in enumerate(configs)}
            W = {(a, b): all_weights(a, b, configs, U, s, eta) for a in range(d) for b in range(d)}
            rho = normalize(np.array([[W[(a, b)].sum() for b in range(d)] for a in range(d)]))

            keeps, sizes, size_of = {}, [], {}
            for a in range(d):                                                  # per-slice threshold selection
                for b in range(d):
                    w = W[(a, b)]
                    if N == 1:
                        keeps[(a, b)] = []
                        continue
                    seen = sorted(index[p] for p in metropolis_configs(lambda p: w[index[p]], 2 * (N - 1), d,
                                                                       n_sample, seed=10 * a + b + 1))
                    frac = marginals(w, configs, d, np.array(seen))
                    frac = frac / frac.sum(axis=1, keepdims=True)
                    keeps[(a, b)] = [set(np.flatnonzero(frac[k] >= zeta)) | {int(np.argmax(frac[k]))}
                                     for k in range(2 * (N - 1))]
            L = np.zeros((d, d), complex)
            for a in range(d):                                                  # (a,b) and (b,a) share a mirror set
                for b in range(d):
                    kept = [keeps[(a, b)][k] | mirror(keeps[(b, a)])[k] for k in range(2 * (N - 1))] \
                        if N > 1 else []
                    mask = np.ones(len(configs), bool)
                    for k, allowed in enumerate(kept):
                        mask &= np.isin(configs[:, k], list(allowed))
                    L[a, b] = W[(a, b)][mask].sum()
                    sizes.append(int(mask.sum()))
                    size_of[(a, b)] = int(mask.sum())
            rho_S = normalize(L)

            # on-the-fly filtering: the path sum grown slice by slice and pruned at every slice at the one
            # fixed cutoff, and mirrored between (a,b) and (b,a) exactly as the Monte-Carlo sets are.
            # The cutoff was formerly bisected to the Monte-Carlo set size so the two selections could be
            # read at equal cost; the figure no longer shows the Monte-Carlo readout, so that target is not
            # on display and the filter is left with the parameter it actually has.
            mir = {} if N == 1 else {tuple(c): i for i, c in enumerate(mirror_configs(configs))}
            filt, evals = {}, 0
            if N == 1:
                filt, evals = {(a, b): {0} for a in range(d) for b in range(d)}, 1
            else:
                for a in range(d):
                    for b in range(d):
                        cfg, _, ne = ofpf_filtered(a, b, N, U, s, eta, d, cut)
                        filt[(a, b)] = {index[tuple(c)] for c in cfg}
                        evals += ne
            L_F, sizes_F = np.zeros((d, d), complex), []
            for a in range(d):
                for b in range(d):
                    keep = filt[(a, b)] | {mir[tuple(configs[q])] for q in filt[(b, a)]} if N > 1 else {0}
                    L_F[a, b] = W[(a, b)][sorted(keep)].sum()
                    sizes_F.append(len(keep))
            rho_F = normalize(L_F)

            # the filter's set reduced per slice into a product set: the |w| marginal of each slice over
            # the surviving paths, keeping every state whose share reaches zeta and the heaviest state.
            # This is what the circuit prepares -- one block per slice -- where the raw filter set is not a
            # product and so bounds a selection rather than a readout. The reduction is not a closure, so
            # the product set can be smaller than the set it came from, and where the marginals stay near
            # uniform it is the whole space and no saving is left.
            L_B, sizes_B = np.zeros((d, d), complex), []
            for a in range(d):
                for b in range(d):
                    sub = sorted(filt[(a, b)] | {mir[tuple(configs[q])] for q in filt[(b, a)]}) if N > 1 else [0]
                    w = W[(a, b)]
                    if N == 1 or not sub:
                        L_B[a, b] = w[sub].sum() if sub else 0.0
                        sizes_B.append(len(sub))
                        continue
                    frac = np.array([[np.abs(w[sub][configs[sub, k] == v]).sum() for v in range(d)]
                                     for k in range(2 * (N - 1))])
                    frac = frac / np.where(frac.sum(axis=1, keepdims=True) > 0,
                                           frac.sum(axis=1, keepdims=True), 1)
                    kept = [set(np.flatnonzero(frac[k] >= zeta)) | {int(np.argmax(frac[k]))}
                            for k in range(2 * (N - 1))]
                    mask = np.ones(len(configs), bool)
                    for k, allowed in enumerate(kept):
                        mask &= np.isin(configs[:, k], list(allowed))
                    L_B[a, b] = w[mask].sum()
                    sizes_B.append(int(mask.sum()))
            rho_B = normalize(L_B)

            # the filter's own ideal: the R largest weights, which needs every one of the D weights scored
            top = {(a, b): set(np.argsort(-np.abs(W[(a, b)]))[:size_of[(a, b)]].tolist())
                   for a in range(d) for b in range(d)}
            L_T = np.zeros((d, d), complex)
            for a in range(d):
                for b in range(d):
                    keep = set(top[(a, b)]) | {mir[tuple(configs[q])] for q in top[(b, a)]} if N > 1 else {0}
                    L_T[a, b] = W[(a, b)][sorted(keep)].sum()
            rho_T = normalize(L_T)
            row = dict(system=system['name'], N=N, t=N * system['dt'], d=d, zeta=zeta,
                       R=float(np.mean(sizes)), D=len(configs),
                       exact=[float(np.real(rho[0, 0])), float(np.real(rho[0, 1])), float(np.imag(rho[0, 1]))],
                       selected=[float(np.real(rho_S[0, 0])), float(np.real(rho_S[0, 1])),
                                 float(np.imag(rho_S[0, 1]))],
                       ofpf=[float(np.real(rho_F[0, 0])), float(np.real(rho_F[0, 1])),
                             float(np.imag(rho_F[0, 1]))],
                       ofpf_rank=[float(np.real(rho_T[0, 0])), float(np.real(rho_T[0, 1])),
                                  float(np.imag(rho_T[0, 1]))],
                       ofpf_box=[float(np.real(rho_B[0, 0])), float(np.real(rho_B[0, 1])),
                                 float(np.imag(rho_B[0, 1]))],
                       R_ofpf=float(np.mean(sizes_F)), R_ofpf_box=float(np.mean(sizes_B)),
                       ofpf_cutoff=float(cut),
                       ofpf_eval=float(evals),
                       dev_fpf=(float(np.abs(rho - run_fpf(system, N)).max())
                                if d ** (2 * (N + 1)) <= FPF_MAX_PATHS else None))
            rows.append(row)
            print(row, flush=True)
            json.dump(rows, open(path, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

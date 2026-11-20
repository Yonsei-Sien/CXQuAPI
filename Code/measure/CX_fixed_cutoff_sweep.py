"""The results readout at a cutoff fixed once, instead of one bisected per system and step.

The figure shows only filter-based readouts, so matching the filter's set size to a Monte-Carlo selection
that is not plotted buys nothing and leaves the cutoff as one derived number per system and step. A fixed
absolute threshold on |prefix weight| is the filter's own parameter and the convention QUAPI.FPF uses, and
it makes the number of surviving paths an output rather than a target. This sweeps candidate values and
reports, for each, what the filter keeps and how far both readouts land from the exact path sum.

  per path  : the surviving paths read directly (green in the results figure)
  per slice : those paths reduced to the states each slice keeps above zeta (blue), a product over slices
  [Input]  argv: zeta (default 0.1)
  [Output] Data/<System>/Readout/fixed_cutoff_sweep.json, and a printed table per candidate cutoff
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, h_bar
from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.CX_mc_readout_N6 import all_weights, normalize
from lib.cx_ofpf_lib import ofpf_filtered
from lib.paths import data_path

PROPERTY = 'Readout'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = ((tls(), 8), (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), 8), (sim2001(0.72), 6))
CUTOFFS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def mirror_configs(configs):
    out = configs.copy()
    out[:, 0::2], out[:, 1::2] = configs[:, 1::2], configs[:, 0::2]
    return out


# zeta-thresholded product set from the |w| marginals of a subset: what one block per slice prepares
def box_mask(w, configs, d, n_slice, sub, zeta):
    frac = np.array([[np.abs(w[sub][configs[sub, k] == v]).sum() for v in range(d)] for k in range(n_slice)])
    tot = frac.sum(axis=1, keepdims=True)
    frac = frac / np.where(tot > 0, tot, 1)
    kept = [set(np.flatnonzero(frac[k] >= zeta)) | {int(np.argmax(frac[k]))} for k in range(n_slice)]
    mask = np.ones(len(configs), bool)
    for k, allowed in enumerate(kept):
        mask &= np.isin(configs[:, k], list(allowed))
    return mask


def main():
    zeta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    rows = []
    for system, N_max in CASES:
        d = system['N_state']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        for N in range(2, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            n_site = 2 * (N - 1)
            configs = np.array(list(itertools.product(range(d), repeat=n_site)), dtype=int)
            index = {tuple(c): i for i, c in enumerate(configs)}
            mir = {tuple(c): i for i, c in enumerate(mirror_configs(configs))}
            W = {(a, b): all_weights(a, b, configs, U, s, eta) for a in range(d) for b in range(d)}
            exact = normalize(np.array([[W[(a, b)].sum() for b in range(d)] for a in range(d)]))
            ex = np.array([np.real(exact[0, 0]), np.real(exact[0, 1]), np.imag(exact[0, 1])])

            for cut in CUTOFFS:
                filt, n_eval = {}, 0
                for a in range(d):
                    for b in range(d):
                        cfg, _, ne = ofpf_filtered(a, b, N, U, s, eta, d, cut)
                        filt[(a, b)] = {index[tuple(c)] for c in cfg}
                        n_eval += ne
                L_F, L_B, nF, nB = np.zeros((d, d), complex), np.zeros((d, d), complex), [], []
                for a in range(d):
                    for b in range(d):
                        sub = sorted(filt[(a, b)] | {mir[tuple(configs[q])] for q in filt[(b, a)]})
                        w = W[(a, b)]
                        L_F[a, b] = w[sub].sum() if sub else 0.0
                        nF.append(len(sub))
                        if not sub:
                            nB.append(0)
                            continue
                        m = box_mask(w, configs, d, n_site, np.array(sub), zeta)
                        L_B[a, b] = w[m].sum()
                        nB.append(int(m.sum()))
                def dev(L):
                    r = normalize(L)
                    return float(np.abs(np.array([np.real(r[0, 0]), np.real(r[0, 1]), np.imag(r[0, 1])]) - ex).max())
                rows.append(dict(system=system['name'], N=N, D=len(configs), cutoff=cut,
                                 R_path=float(np.mean(nF)), R_slice=float(np.mean(nB)),
                                 dev_path=dev(L_F), dev_slice=dev(L_B), n_eval=float(n_eval)))
                print(f"{system['name']:12s} N={N} D={len(configs):6d} cut={cut:.0e} "
                      f"R_path={np.mean(nF):8.1f} ({np.mean(nF)/len(configs):6.1%})  "
                      f"R_slice={np.mean(nB):8.1f} ({np.mean(nB)/len(configs):6.1%})  "
                      f"dev_path={rows[-1]['dev_path']:.2e}  dev_slice={rows[-1]['dev_slice']:.2e}", flush=True)
            json.dump(rows, open(os.path.join(ROOT_DIR(), 'fixed_cutoff_sweep.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Which paths should the selected readout put in its computational basis?

The readout measures one amplitude, <u_S|psi> with |u_S> uniform over a set S of intermediate configurations, and
the circuit never learns how S was chosen, so any path-selection rule is a drop-in.  Three rules are compared on
their own accuracy-cost front, each swept over its own control parameter:

  mc      : the configurations a Metropolis walk actually visits, used as they come
            (Sim & Makri, CPL 249, 224 (1996)); swept over the number of samples
  product : the present scheme -- the same walk reduced to per-slice marginals, values above zeta kept, their
            Cartesian product taken; swept over zeta
  ofpf    : the configurations whose weight clears an absolute cutoff, as the on-the-fly filter keeps them
            (Sim, JCP 115, 4450 (2001)); swept over the set size the cutoff produces
  box     : the smallest product set containing those survivors -- the OFPF support, made cheap to prepare again
  marg    : exact |w| marginals per slice, values above zeta kept, their product taken; the present rule with the
            preliminary walk removed, so its accuracy is no longer capped by what a finite walk happened to visit

The estimator is  A_hat = sqrt(R) <u_S|psi> = sum_{i in S} w_i, so one element's readout weight is |sum_S w|^2 / R
and the whole RDM costs sum_ab 1/weight_ab shots: the hardest element sets the bill, not the average one.  Error
is reported the same way, as ||rho_hat - rho||_F/||rho||_F over the d^2 elements rather than element by element.
Preparing |u_S> costs one small preparation per register when S is a product, and otherwise one rotation per node
of its prefix tree, which is the third quantity reported: the rules that keep slice-slice correlations buy their
accuracy with state preparation.
  [Input]  argv: N (default 5); the driver in Running/ sets SYSTEM and SYSTEMS
  [Output] Data/<System>/Readout/readout_rules.json : one row per (rule, control parameter), carrying the
           basis size R it produced, the error of the whole reduced density matrix, the shots it saves over
           reading the full sum, and the rotations its preparation costs. FIGURE is off by default; the
           supplementary table is read straight off the json.
"""
import contextlib
import io
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, metropolis_configs, h_bar
from lib.cx_systems_lib import tls, sim2001, ohmic
from lib.CX_mc_readout_N6 import all_weights
from lib.CX_mc_readout_product import marginals
from lib.paths import data_path

PROPERTY = 'Readout'
SYSTEM = 'Common'      # the driver in Running/ sets this
FIGURE = False         # the three-panel figure needs matplotlib and the nature style module
SYSTEMS = (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), sim2001(0.72))


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)

N_SAMPLE = (100, 300, 1000, 3000, 10000, 30000)                        # mc:      walk length
ZETA     = (0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.003, 0.001)             # product: marginal cutoff
FRAC     = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)   # ofpf:    set size as a fraction of D

# categorical slots 1-3 of the validated palette (see CX_figures_nature), each with its own marker and dashes
STYLE = {'mc':      dict(label='MC paths',          color='#2a78d6', marker='o', ls='-'),
         'product': dict(label='MC + threshold',    color='#eb6834', marker='s', ls='--'),
         'ofpf':    dict(label='OFPF',              color='#1baf7a', marker='^', ls='-.'),
         'box':     dict(label='OFPF, made product', color='#eda100', marker='D', ls=(0, (3, 1, 1, 1))),
         'marg':    dict(label='exact marginals',   color='#8d5fc9', marker='v', ls=(0, (1, 1)))}
# marg is computed but not drawn: it falls on top of product to three digits, which is itself the finding --
# the preliminary walk is not what limits the present rule, the product form is
ORDER = ('mc', 'product', 'ofpf', 'box')


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# S = the distinct configurations the walk visits, used as they come
def mc_set(w, configs, index, d, n_slice, n_sample, seed):
    seen = quiet(metropolis_configs, lambda p: w[index[p]], n_slice, d, n_sample, seed=seed)
    return np.array(sorted(index[p] for p in seen))


# S = product of the per-slice values whose marginal clears zeta (the present scheme)
def product_set(w, configs, index, d, n_slice, zeta, seed):
    seen = quiet(metropolis_configs, lambda p: w[index[p]], n_slice, d, 3000, seed=seed)
    frac = marginals(w, configs, d, np.array(sorted(index[p] for p in seen)))
    frac = frac / frac.sum(axis=1, keepdims=True)
    kept = [set(np.flatnonzero(frac[k] >= zeta)) | {int(np.argmax(frac[k]))} for k in range(n_slice)]
    mask = np.ones(len(configs), bool)
    for k, allowed in enumerate(kept):
        mask &= np.isin(configs[:, k], list(allowed))
    return np.flatnonzero(mask)


# S = the R configurations of largest |w|, which is what an absolute cutoff on the weight keeps
def ofpf_set(w, R):
    return np.argsort(-np.abs(w))[:R]


# Smallest product set containing a given set: each slice keeps the values that appear in it
def box_of(configs, S, n_slice):
    mask = np.ones(len(configs), bool)
    for k in range(n_slice):
        mask &= np.isin(configs[:, k], np.unique(configs[S, k]))
    return np.flatnonzero(mask)


# S = product of the per-slice values whose exact |w| marginal clears zeta -- no preliminary walk anywhere
def marg_set(w, configs, d, n_slice, zeta):
    frac = np.array([[np.abs(w[configs[:, k] == v]).sum() for v in range(d)] for k in range(n_slice)])
    frac = frac / frac.sum(axis=1, keepdims=True)
    kept = [set(np.flatnonzero(frac[k] >= zeta)) | {int(np.argmax(frac[k]))} for k in range(n_slice)]
    mask = np.ones(len(configs), bool)
    for k, allowed in enumerate(kept):
        mask &= np.isin(configs[:, k], list(allowed))
    return np.flatnonzero(mask)


# Rotations a sequential preparation of |u_S> needs: one per distinct prefix of the kept configurations
def tree_nodes(configs, S):
    pre = set()
    for c in configs[S]:
        t = ()
        for v in c:
            t += (int(v),); pre.add(t)
    return len(pre)


# Per-element readout weight |<u_S|psi>|^2 (up to the common circuit factor) and the element the readout returns
#   shots for one element go as 1/|amp|^2, so the whole RDM costs sum_ab 1/|amp_ab|^2 -- the hard element rules
def element(w, S, D):
    A_hat = w[S].sum()
    return abs(A_hat) ** 2 / len(S), A_hat


# Aggregate a rule over the d^2 elements: shots relative to reading the full sum, and the error of the whole RDM
#   [Input]  amps, fulls : |amp|^2 and A_hat per element, for the rule and for the full path sum
#   [Output] gain : (total shots of the full sum) / (total shots of the rule);  err : ||rho_hat - rho||_F/||rho||_F
def aggregate(amp_S, A_S, amp_full, A_full):
    shots_S    = sum(1.0 / a for a in amp_S)
    shots_full = sum(1.0 / a for a in amp_full)
    err = np.sqrt(sum(abs(h - f) ** 2 for h, f in zip(A_S, A_full))) / np.sqrt(sum(abs(f) ** 2 for f in A_full))
    return shots_full / shots_S, float(err)


def sweep(system, N):
    d, n_slice = system['N_state'], 2 * (N - 1)
    U   = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
    s   = np.real(np.asarray(system['S_c'], dtype=complex))
    eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
    configs = np.array(list(itertools.product(range(d), repeat=n_slice)))
    index, D = {tuple(c): i for i, c in enumerate(configs)}, len(configs)
    weights = {(a, b): all_weights(a, b, configs, U, s, eta) for a in range(d) for b in range(d)}

    keys = sorted(weights)
    full = [element(weights[k], np.arange(D), D) for k in keys]                # reading the whole path sum
    amp_full, A_full = [f[0] for f in full], [f[1] for f in full]

    # the rule's own sets, plus the OFPF set at the same R, so the two are compared at equal basis size
    rows = []
    for rule, params in (('mc', N_SAMPLE), ('product', ZETA), ('ofpf', FRAC), ('box', FRAC), ('marg', ZETA)):
        for p in params:
            sets = []
            for k in keys:
                w, seed = weights[k], 10 * k[0] + k[1] + 1
                if rule == 'mc':
                    sets.append(mc_set(w, configs, index, d, n_slice, int(p), seed))
                elif rule == 'product':
                    sets.append(product_set(w, configs, index, d, n_slice, p, seed))
                elif rule == 'marg':
                    sets.append(marg_set(w, configs, d, n_slice, p))
                elif rule == 'box':
                    sets.append(box_of(configs, ofpf_set(w, max(1, int(round(p * D)))), n_slice))
                else:
                    sets.append(ofpf_set(w, max(1, int(round(p * D)))))
            vals = [element(weights[k], S, D) for k, S in zip(keys, sets)]
            gain, err = aggregate([v[0] for v in vals], [v[1] for v in vals], amp_full, A_full)
            # How much of the error one starved element carries. A product set apportions its size by each
            # element's own marginals, so an element whose marginals are peaked keeps a single configuration
            # however low the cutoff goes, and no rule can do better at R = 1: that element then floors the
            # whole matrix. Reported in the same norm as err, so the two are directly comparable.
            j = int(np.argmin([len(S) for S in sets]))
            err_small = abs(vals[j][1] - A_full[j]) / np.sqrt(sum(abs(f) ** 2 for f in A_full))
            product_form = rule in ('product', 'box', 'marg')                  # one small preparation per register
            prep = np.mean([n_slice * d if product_form else tree_nodes(configs, S) for S in sets])

            matched = [ofpf_set(weights[k], len(S)) for k, S in zip(keys, sets)]       # OFPF at the same R
            mvals = [element(weights[k], S, D) for k, S in zip(keys, matched)]
            m_gain, m_err = aggregate([v[0] for v in mvals], [v[1] for v in mvals], amp_full, A_full)
            # the matched set is never a product, so its preparation is its prefix tree, as for any
            # correlated set: this is what lets the table report both rules cost at one and the same R
            m_prep = np.mean([tree_nodes(configs, S) for S in matched])

            row = dict(system=system['name'], N=N, d=d, D=D, rule=rule, param=float(p),
                       R=float(np.mean([len(S) for S in sets])), R_min=int(min(len(S) for S in sets)),
                       err_min_element=float(err_small),
                       err=err, gain=float(gain), prep=float(prep),
                       err_ofpf_matched=m_err, gain_ofpf_matched=float(m_gain), prep_ofpf_matched=float(m_prep))
            rows.append(row)
            print(f"  {rule:>8s} {p:<8g} R {row['R']:8.0f}/{D}   err {err:.3e}   gain {gain:8.2f}   "
                  f"prep {prep:7.0f}   | OFPF at same R: err {m_err:.3e}  gain {m_gain:8.2f}", flush=True)
    return rows


def r_int(row):
    return int(round(row['R']))


def figure(rows, system_name, N):
    import matplotlib.pyplot as plt
    from lib.figure_style import MM, DOUBLE, MUTED, bare, panel, save
    data = [r for r in rows if r['system'] == system_name]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 52 * MM))

    for ax, letter in zip(axes, 'abc'):
        bare(ax); panel(ax, letter, dx=-0.30, dy=1.12)
        ax.set_xscale('log'); ax.set_yscale('log')

    for rule in ORDER:
        st  = STYLE[rule]
        sub = sorted((r for r in data if r['rule'] == rule), key=lambda r: r['R'])
        kw  = dict(color=st['color'], marker=st['marker'], ls=st['ls'], ms=3.0, lw=0.9, label=st['label'])
        vis = [r for r in sub if r['err'] > 1e-11]                  # below this the set reproduces the full sum
        axes[0].plot([r['R'] for r in vis], [r['err'] for r in vis], **kw)
        front = sorted(vis, key=lambda r: r['err'])
        axes[1].plot([r['err'] for r in front], [r['gain'] for r in front], **kw)
        axes[2].plot([r['R'] for r in sub], [r['prep'] for r in sub], **kw)

    D = data[0]['D']
    axes[0].set_xlabel('paths in the readout basis, $R$')
    axes[0].set_ylabel(r'RDM error, $\|\hat\rho-\rho\|_F/\|\rho\|_F$')
    axes[0].axvline(D, color=MUTED, lw=0.5, ls=':')
    axes[0].text(D * 0.88, axes[0].get_ylim()[1] * 0.4, 'all $D$ paths', fontsize=5, color=MUTED,
                 rotation=90, ha='right', va='top')
    axes[0].legend(loc='lower left', fontsize=5.5, handletextpad=0.4, labelspacing=0.3)

    axes[1].set_xlabel(r'RDM error, $\|\hat\rho-\rho\|_F/\|\rho\|_F$')
    axes[1].set_ylabel('shots saved over the full sum')
    axes[1].invert_xaxis()
    axes[1].axhline(1.0, color=MUTED, lw=0.5, ls=':')
    axes[1].text(axes[1].get_xlim()[1], 1.15, 'no saving', fontsize=5, color=MUTED, ha='left', va='bottom')

    # the present operating point, and the best any rule reaches: both read straight off the front
    here = next(r for r in data if r['rule'] == 'product' and r['param'] == 0.1)
    best = max((r for r in data if r['rule'] == 'ofpf' and r['err'] > 1e-11), key=lambda r: r['gain'])
    axes[1].set_ylim(0.55, best['gain'] * 6)
    for r, text, xf, yf in ((here, f"present scheme\n($\\zeta=0.1$, $R={r_int(here)}$)", 0.09, 0.40),
                            (best, f"best OFPF cut\n($R={r_int(best)}$)", 1.0, 3.4)):
        axes[1].annotate(text, xy=(r['err'], r['gain']), xytext=(r['err'] * xf, r['gain'] * yf),
                         fontsize=5, color=MUTED, ha='center', va='center',
                         arrowprops=dict(arrowstyle='-', lw=0.4, color=MUTED, shrinkA=2, shrinkB=2))

    axes[2].set_xlabel('paths in the readout basis, $R$')
    axes[2].set_ylabel('rotations to prepare $|u_S\\rangle$')

    fig.text(0.5, 0.075, f'{system_name}, $N={N}$, $D={D}$ paths in all. Each curve is one selection rule swept '
                         f'over its own cutoff.', ha='center', fontsize=5.5, color=MUTED)
    fig.text(0.5, 0.020, 'Shots are summed over the $d^2$ elements and the error is taken over the whole matrix, '
                         'so the hardest element sets both.', ha='center', fontsize=5.5, color=MUTED)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.92, bottom=0.30, wspace=0.40)
    save(fig, 'sien_readout_rules')


def main(N=None):
    N = int(sys.argv[1]) if N is None and len(sys.argv) > 1 else (5 if N is None else N)
    os.makedirs(ROOT_DIR(), exist_ok=True)
    rows = []
    for system in SYSTEMS:
        print(f"{system['name']}  N={N}  d={system['N_state']}", flush=True)
        rows += sweep(system, N)
    json.dump(rows, open(os.path.join(ROOT_DIR(), 'readout_rules.json'), 'w'), indent=1)
    if FIGURE:
        figure(rows, SYSTEMS[-1]['name'], N)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Per-slice marginals of the path weight, which is what a product-form readout layer selects on.

A product-form layer keeps, in every time slice, the states whose weight marginal

    m_k(s) = sum_{p : p_k = s} |w_ab(p)|

is above a threshold, and reads out the uniform state over the product of the kept sets. The layer then
stays a product of one small preparation per slice -- the cost of the Hadamard layer it replaces --
unlike a general selected set. `marginals` is that m_k(s), and the selection rule itself lives with the
measurement that applies it (measure/CX_layer_hadamard.py, measure/CX_readout_rules.py).

The threshold sweep this module used to carry (written to Data/TP/mc_readout_product_N{N}.json) was
removed on 2026-11-18 together with the sweep in CX_mc_readout_N6.py: nothing read that file.

  [Input]  w, configs : path weights and the configurations they belong to
           d          : number of states per slice
           visited    : row indices of a Metropolis walk, or None for the exact marginals
  [Output] (2(N-1), d) array of marginal weights, one row per slice
"""
import numpy as np


# Per-slice marginals of |w| over the configurations visited (or over all of them if visited is None)
def marginals(w, configs, d, visited=None):
    rows = configs if visited is None else configs[visited]
    vals = np.abs(w) if visited is None else np.abs(w[visited])
    return np.array([[vals[rows[:, k] == s].sum() for s in range(d)] for k in range(configs.shape[1])])

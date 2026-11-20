"""Does one number predict the group size the influence functional needs?

The grouped construction is exact once a family of groups reaches alpha = 1, and the smallest such
group g* is what a circuit pays for. Neither the bath cut-off w_c nor the temperature beta predicts
g* on its own; both scans collapse onto one trend against the diagonal dominance of the blip matrix

    r = min_k  M_kk / sum_{k' != k} |M_kk'| ,

one number for how much of the influence functional lives on a single time slice rather than between
slices. r >= 1 is the classical diagonal-dominance criterion, under which pairs of slices already
suffice; below it the lattice condition's sign structure takes over and the trend loosens.

Restored from the Attic (see Trials.md): the file it writes is still read by Analysis/Fig2.ipynb,
so the measurement has to stay reproducible. The figure it used to draw now lives in that notebook,
and the two scan lengths are module constants rather than argv, so a driver cannot silently shorten
the scan the way CX_walsh_vs_lp_opt.py once did.

  [Input]  N_G : time slices of the M whose smallest exact group is measured
           N_M : time slices of the M whose dominance r is measured
           BASE / W_C_VALUES / BETA_VALUES : the ohmic TLS bath and the two scans over it
  [Output] Data/Common/Bath/fig3_dominance.json : one row per scan point with
           (key, value, dominance, g, cx, alpha_eta)
"""
import json
import os

from lib.cx_systems_lib import ohmic, tls
from lib.paths import data_path
from measure.CX_coupling_sweep import LAT_TLS, M_matrix, smallest_exact_group

SYSTEM   = 'Common'          # the two scans are over baths, not over one named system
PROPERTY = 'Bath'
N_G      = 5                 # g* is measured on the N = 5 blip matrix, where g = 2..6 is reachable
N_M      = 7                 # r is measured on the longer window, where the bath tail is included
BASE     = dict(xi=0.1, w_c=2.5, beta=5.0)   # the weakly damped TLS both scans start from
W_C_VALUES  = (1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0)    # bath cut-off, in units of 1/dt
BETA_VALUES = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)    # inverse temperature


def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)


# r = min_k M_kk / sum_{k' != k} |M_kk'| : diagonal dominance of the blip matrix
#   [Input]  M : blip matrix of the influence functional
#   [Output] r : 1 and above means every row is dominated by its own slice
def dominance(M):
    n = len(M)
    return min(M[k, k] / sum(abs(M[k, j]) for j in range(n) if j != k) for k in range(n))


# One scan: move a single bath parameter and record (r, g*) at each point.
#   [Input]  key : 'w_c' or 'beta', the parameter being moved
#            values : the points of the scan
#   [Output] rows : one dict per point, as written to the json
def scan(key, values):
    rows = []
    for v in values:
        p = dict(BASE)
        p[key] = float(v)
        system = dict(tls(), sd=ohmic(p['xi'], p['w_c']), beta=p['beta'])
        # g* = smallest window size whose LP reaches alpha = 1 on the blip lattice
        g, cx, a_pair, _ = smallest_exact_group(M_matrix(system, N_G), LAT_TLS, 1, N_G)
        r = float(dominance(M_matrix(system, N_M)))
        rows.append(dict(key=key, value=float(v), dominance=r, g=g, cx=cx, alpha_eta=a_pair))
        print(f'  {key}={v:>6.2f}  r = {r:.4f}  g* = {g}  CNOT = {cx}', flush=True)
    return rows


def main():
    print('cut-off scan', flush=True)
    cut = scan('w_c', W_C_VALUES)
    print('temperature scan', flush=True)
    tem = scan('beta', BETA_VALUES)
    out = os.path.join(ROOT_DIR(), 'fig3_dominance.json')
    json.dump(cut + tem, open(out, 'w'), indent=1)
    print('written:', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

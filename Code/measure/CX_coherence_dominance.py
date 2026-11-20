"""Does the path sum of a COHERENCE concentrate the way a population's does?

The body's readout claim rests on concentration of the path sum, not on the size of the
element, and the concentration was measured on rho_11 alone.  A coherence is the element
strong coupling destroys, so if the claim is element-dependent this is where it breaks.
Same two numbers as CX_path_dominance.py, now for (a, b) = (0, 0) and (0, 1).

  [Input]  argv[1] : largest N (default 8, the longest propagation Supplementary Table~S8 quotes)
  [Output] Data/CouplingSweep/coherence_dominance.json, stdout table
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

from pyqake.ansatz.CXQUAPI import alpha_generator, h_bar
from lib.cx_systems_lib import tls, ohmic
from lib.CX_mc_readout_N6 import all_weights
from measure.CX_path_dominance import participation, paths_for
from lib.paths import data_path

PROPERTY = 'Bath'
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
SYSTEMS = (tls(), dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2))


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(ROOT_DIR(), exist_ok=True)
    rows = []
    for system in SYSTEMS:
        d = system['N_state']
        U = expm(-1j * np.asarray(system['H_c']) * system['dt'] / h_bar)
        s = np.real(np.asarray(system['S_c'], dtype=complex))
        print(f"=== {system['name']} ===")
        print(f"{'N':>3s}{'D':>7s}"
              f"{'PR(pop)':>9s}{'R2(pop)':>8s}{'R3(pop)':>8s}"
              f"{'PR(coh)':>9s}{'R2(coh)':>8s}{'R3(coh)':>8s}"
              f"{'|sum|pop':>11s}{'|sum|coh':>11s}")
        for N in range(2, N_max + 1):
            eta = alpha_generator(system['dt'], N, N, system['beta'], system['sd'])
            configs = np.array(list(itertools.product(range(d), repeat=2 * (N - 1))))
            out = {}
            for tag, (a, b) in (('pop', (0, 0)), ('coh', (0, 1))):
                w = all_weights(a, b, configs, U, s, eta)
                out[tag] = (participation(w), paths_for(w, 1e-2), paths_for(w, 1e-3),
                            abs(complex(w.sum())))
            rows.append(dict(system=system['name'], N=N, D=len(configs),
                             pr_pop=out['pop'][0], r2_pop=out['pop'][1], r3_pop=out['pop'][2],
                             sum_pop=out['pop'][3],
                             pr_coh=out['coh'][0], r2_coh=out['coh'][1], r3_coh=out['coh'][2],
                             sum_coh=out['coh'][3]))
            print(f'{N:>3d}{len(configs):>7d}'
                  f"{out['pop'][0]:>9.2f}{out['pop'][1]:>8d}{out['pop'][2]:>8d}"
                  f"{out['coh'][0]:>9.2f}{out['coh'][1]:>8d}{out['coh'][2]:>8d}"
                  f"{out['pop'][3]:>11.6f}{out['coh'][3]:>11.6f}", flush=True)
        print()
    json.dump(rows, open(os.path.join(ROOT_DIR(), 'coherence_dominance.json'), 'w'), indent=1)
    print('written')


if __name__ == '__main__':
    sys.exit(main())

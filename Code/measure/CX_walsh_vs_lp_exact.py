"""Walsh against the linear program at equal exactness: both at alpha = 1.

CX_walsh_vs_lp.py runs the program at its default pairs, which is exact in the dynamics but leaves
alpha > 1 for a strongly coupled bath, so the shot cost differs between the two constructions and the
gate counts are not comparable. Here the group size is raised until the program returns alpha = 1, and
only then is the circuit counted. The fused Walsh construction is alpha = 1 by its nature, so the two
are then paying the same price in shots and the comparison is of gates alone.
  [Input]  argv: N_max (default 6), alpha tolerance (default 1e-9)
  [Output] Data/<System>/Walsh/walsh_vs_lp_exact.json
"""
import contextlib
import io as _io
import json
import os
import sys
import time

import numpy as np
from qiskit import transpile

from pyqake.ansatz.CXQUAPI import CXQUAPI, COUNT_BASIS
from lib.paths import data_path
from lib.cx_systems_lib import tls, sim2001, ohmic

PROPERTY = 'Walsh'
N_MAX    = 6           # longest memory scanned; the driver sets it, argv still overrides
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = ((tls(), 'TLS'), (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), 'TLS-strong'),
         (sim2001(0.72), 'Sim2001'))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(_io.StringIO()):
        return fn(*a, **k)


def make(system, N, IF, g=None):
    kw = dict(N_memory=N, N_state=system['N_state'], beta=system['beta'], IF=IF)
    if g is not None:
        kw['IF_setting'] = {'group_size': g}
    obj = CXQUAPI(system['H_q'], system['S_q'], system['dt'], N, **kw)
    obj.spectral_density = system['sd']
    return obj


# Smallest group for which the program returns alpha = 1, and the circuit it builds there
def smallest_exact(system, N, tol):
    for g in range(2, N + 2):
        obj = make(system, N, 'lp', g)
        circ = quiet(obj.gen_propagation_circuit)
        if float(obj.alpha_lp) < 1.0 + tol:
            return g, obj, circ
    return None, None, None


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else N_MAX
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-9
    rows, path = [], os.path.join(ROOT_DIR(), 'walsh_vs_lp_exact.json')
    for system, name in CASES:
        print(f"\n=== {name}  d={system['N_state']}   (both at alpha = 1)", flush=True)
        print(f"{'N':>2} {'g*':>3} {'cx_lp':>8} {'cx_walsh':>10} {'ratio':>7} "
              f"{'qb_lp':>6} {'qb_walsh':>9} {'t_lp':>7} {'t_walsh':>8}", flush=True)
        for N in range(2, N_max + 1):
            try:
                t0 = time.time()
                g, obj, circ = smallest_exact(system, N, tol)
                t_lp = time.time() - t0
                if g is None:
                    print(f"{N:>2}   no group reached alpha = 1", flush=True)
                    continue
                cx_lp = int(transpile(circ, basis_gates=COUNT_BASIS,
                                      optimization_level=0).count_ops().get('cx', 0))
                t1 = time.time()
                ow = make(system, N, 'walsh_fourier')
                cw = quiet(ow.gen_propagation_circuit)
                t_w = time.time() - t1
                cx_w = int(transpile(cw, basis_gates=COUNT_BASIS,
                                     optimization_level=0).count_ops().get('cx', 0))
                rec = dict(system=name, N=N, g_star=g, cx_lp=cx_lp, cx_walsh=cx_w,
                           ratio=cx_w / max(cx_lp, 1), qubits_lp=int(circ.num_qubits),
                           qubits_walsh=int(cw.num_qubits), alpha_lp=float(obj.alpha_lp),
                           t_lp=t_lp, t_walsh=t_w)
                print(f"{N:>2} {g:>3} {cx_lp:>8} {cx_w:>10} {rec['ratio']:>7.1f} "
                      f"{rec['qubits_lp']:>6} {rec['qubits_walsh']:>9} {t_lp:>7.1f} {t_w:>8.1f}",
                      flush=True)
            except Exception as exc:
                rec = dict(system=name, N=N, failed=f'{type(exc).__name__}: {exc}'[:160])
                print(f"{N:>2}   {rec['failed']}", flush=True)
            rows.append(rec)
            json.dump(rows, open(path, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

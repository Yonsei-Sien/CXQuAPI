"""Does the gate-count margin survive compiler optimization?

CX_walsh_vs_lp_exact.py counts gates at optimization_level=0, which is the structural count: it measures
the construction and not the compiler. That is the right quantity to compare two constructions by, but a
reader may reasonably ask whether an optimizing pass closes the gap, since the two build their circuits
out of different primitives -- uniformly controlled rotations on a group for the program, parity ladders
per Walsh string for the fused construction. Here both are recompiled at every optimization level and the
margin is recomputed.

Level 3 runs heavy passes, so it is attempted only while the structural count is below a size at which it
would dominate the run; whatever is skipped is reported as skipped rather than left out.
  [Input]  N_MAX / CEILING module constants, which a driver sets; argv still overrides both
  [Output] Data/<System>/Walsh/walsh_vs_lp_opt.json
"""
import contextlib
import io as _io
import json
import os
import sys
import time

from qiskit import transpile

from pyqake.ansatz.CXQUAPI import CXQUAPI, COUNT_BASIS
from lib.paths import data_path
from lib.cx_systems_lib import tls, sim2001, ohmic

PROPERTY = 'Walsh'
N_MAX    = 6           # longest memory scanned; the driver sets it, argv still overrides
# levels 2-3 are skipped once level 0 is larger than this, since the heavy passes cost minutes on a
# circuit no device could run. 150k clears the N = 6 two-level circuits (115k CNOT) and still leaves
# the three-state N = 5 one (2e8 CNOT) out, which is where the paper's table has its blanks.
CEILING  = 150_000
SYSTEM = 'Common'      # the driver in Running/ sets this; data files under Data/<System>/<Property>/


# Resolved per call rather than at import, so one driver can set SYSTEM before running
def ROOT_DIR():
    return data_path(SYSTEM, PROPERTY)
CASES = ((tls(), 'TLS'), (dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2), 'TLS-strong'),
         (sim2001(0.72), 'Sim2001'))
LEVELS = (0, 1, 2, 3)


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


def smallest_exact(system, N, tol=1e-9):
    for g in range(2, N + 2):
        obj = make(system, N, 'lp', g)
        circ = quiet(obj.gen_propagation_circuit)
        if float(obj.alpha_lp) < 1.0 + tol:
            return g, circ
    return None, None


# CNOTs after each optimization level of `levels`, with the time each pass took beside it
#   [Input]  circ : the circuit to compile;  levels : which optimization levels to run
#   [Output] {level: cx count or None, 't<level>': seconds}
def counts(circ, levels):
    out = {}
    for lv in levels:
        t0 = time.time()
        tc = transpile(circ, basis_gates=COUNT_BASIS, optimization_level=lv)
        out[lv] = int(tc.count_ops().get('cx', 0))
        out[f't{lv}'] = round(time.time() - t0, 1)
    return out


# Which levels a row may use. The two circuits of a row are compared against each other, so they must
# be compiled at the same levels: giving the heavy passes to the smaller one alone would credit it
# with a reduction the other was never offered. The row therefore decides from the LARGER of the two
# level-0 counts, and above the ceiling both stop at level 1.
#   [Input]  cx_lp, cx_walsh : level-0 CNOT counts of the two circuits
#   [Output] the levels both circuits run
def shared_levels(cx_lp, cx_walsh, ceiling):
    return LEVELS if max(cx_lp, cx_walsh) <= ceiling else (0, 1)


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else N_MAX
    ceiling = int(sys.argv[2]) if len(sys.argv) > 2 else CEILING
    rows, path = [], os.path.join(ROOT_DIR(), 'walsh_vs_lp_opt.json')
    for system, name in CASES:
        print(f"\n=== {name}", flush=True)
        print(f"{'N':>2} {'g*':>3} | {'lp L0':>8} {'L1':>8} {'L2':>8} {'L3':>8} "
              f"| {'walsh L0':>9} {'L1':>9} {'L2':>9} {'L3':>9} | {'ratio L0':>9} {'ratio best':>11}",
              flush=True)
        for N in range(2, N_max + 1):
            try:
                g, circ_lp = smallest_exact(system, N)
                if g is None:
                    continue
                circ_w = quiet(make(system, N, 'walsh_fourier').gen_propagation_circuit)
                # levels 0 and 1 first for both: the row's level set depends on the larger count
                c_lp, c_w = counts(circ_lp, (0, 1)), counts(circ_w, (0, 1))
                if shared_levels(c_lp[0], c_w[0], ceiling) == LEVELS:
                    c_lp.update(counts(circ_lp, (2, 3)))
                    c_w.update(counts(circ_w, (2, 3)))
                for c in (c_lp, c_w):               # a level the row did not run reads as None
                    for lv in LEVELS:
                        c.setdefault(lv, None)
                best_lp = min(v for k, v in c_lp.items() if isinstance(k, int) and v is not None)
                best_w = min(v for k, v in c_w.items() if isinstance(k, int) and v is not None)
                rec = dict(system=name, N=N, g_star=g, lp=c_lp, walsh=c_w,
                           ratio_L0=c_w[0] / max(c_lp[0], 1), ratio_best=best_w / max(best_lp, 1))
                f = lambda d, k: '-' if d.get(k) is None else str(d[k])
                print(f"{N:>2} {g:>3} | {f(c_lp,0):>8} {f(c_lp,1):>8} {f(c_lp,2):>8} {f(c_lp,3):>8} "
                      f"| {f(c_w,0):>9} {f(c_w,1):>9} {f(c_w,2):>9} {f(c_w,3):>9} "
                      f"| {rec['ratio_L0']:>9.1f} {rec['ratio_best']:>11.1f}", flush=True)
            except Exception as exc:
                rec = dict(system=name, N=N, failed=f'{type(exc).__name__}: {exc}'[:160])
                print(f"{N:>2}   {rec['failed']}", flush=True)
            rows.append(rec)
            json.dump(rows, open(path, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

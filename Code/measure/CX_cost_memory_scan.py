"""Algorithm 1 with a truncated memory span: the IF keeps only |k - k'| <= K, so the number of groups (and of
ancillas) falls from (N+1)(N+2)/2 to about (K+1)N. Circuit cost and the accuracy it costs.
  [Input]  argv: N_max (default 8)
  [Output] Data/<System>/Cost/cost_memory.json : per N and K, qubits, ancillas, CNOT, depth,
           alpha and the errors; one file per system, which the Shots study then reads
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
from qiskit import transpile

from pyqake.ansatz.CXQUAPI import CXQUAPI, COUNT_BASIS
from lib.cx_systems_lib import tls, ohmic, run_fpf

PROPERTY = 'Cost'       # data lands in Data/<System>/Cost/, one file per system
from lib.paths import data_path

SYSTEMS = (tls(), dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    for system in SYSTEMS:
        for N in range(2, N_max + 1):
            rho_exact = run_fpf(system, N)                                     # full memory reference
            for K in sorted({2, 3, N}):
                if K > N:
                    continue
                g = 2 if system['name'] == 'TLS' else min(K + 1, N + 1)        # windows matched to the memory
                obj = CXQUAPI(system['H_q'], system['S_q'], system['dt'], N, N_memory=K, N_state=2,
                              beta=system['beta'], IF='lp', IF_setting={'group_size': g})
                obj.spectral_density = system['sd']
                circ = quiet(obj.gen_propagation_circuit)
                tc = transpile(circ, basis_gates=COUNT_BASIS, optimization_level=0)
                err = None
                if circ.num_qubits <= 24:
                    err = float(np.abs(quiet(obj.run, s0_f='0', s0_b='0') - rho_exact).max())
                rows.append(dict(system=system['name'], N=N, K=K, g=g, qubits=circ.num_qubits,
                                 ancilla=obj.N_ancilla, groups=len(obj.groups), cx=int(tc.count_ops().get('cx', 0)),
                                 depth=int(tc.depth()), alpha=float(obj.alpha_lp), err_vs_full=err))
                print(rows[-1], flush=True)
                # filed per system, so a partial run still leaves each system's own file consistent
                for tag in sorted({r['system'] for r in rows}):
                    out = os.path.join(data_path(tag, PROPERTY), 'cost_memory.json')
                    json.dump([r for r in rows if r['system'] == tag], open(out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

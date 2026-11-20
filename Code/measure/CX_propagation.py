"""The circuit itself, dumped step by step: the audit trail behind every other circuit number.

Two sweeps, both at full bath memory so that the influence functional is exact:
  dyn : one propagation per N, run and dumped whole. This is the reduced density matrix the
        construction produces, against which the classical path sum is compared.
  res : the same N with every group size g the family allows, plus the fused Walsh dilation,
        dumped without running. These carry the resource counts -- ancillas, CNOTs, alpha.

Nothing in Analysis/ reads either sweep: Table 1 comes from lp_cost_vs_shots.json and Table 2 from
gstar_vs_{N,K}.json, and the registry's read list holds no file of this directory. The sweep is kept
for TLS alone, as the audit trail behind the circuit numbers, at the lengths Data/TLS/Propagation
holds. The run sweep is a full statevector, so d and N set the memory: d = 2 at N = 6 is 35 qubits
(550 GB) and d = 3 at N = 5 is 39 qubits (8.8 TB), which Aer refuses as wider than it can hold.

  U = exp(-i H dt / hbar) per slice, with the influence functional applied as a dilated
  diagonal operator; `IF='lp'` cuts it by the linear program at group size g, `IF='walsh_fourier'`
  fuses the whole register into one.

  [Input]  system : a cx_systems_lib dict (H_q, S_q, dt, beta, sd, N_state)
           N_dyn  : highest N of the run-and-dump sweep
           N_res  : highest N of the resource sweep
  [Output] Data/<System>/Propagation/dyn_N<N>.json
           Data/<System>/Propagation/res_N<N>_lp_g<g>.json, res_N<N>_walsh_g0.json
"""
import sys

from pyqake.ansatz.CXQUAPI import CXQUAPI

from lib.paths import data_path

# the four quantities dumped from a resource run: enough to rebuild a cost table, not the whole state
LEVELS = (0, 1, 2, 3)


def build(system, N, IF='lp', g=None):
    kw = dict(N_memory=N, N_state=system['N_state'], beta=system['beta'], IF=IF)
    if g is not None:
        kw['IF_setting'] = {'group_size': g}
    obj = CXQUAPI(system['H_q'], system['S_q'], system['dt'], N, **kw)
    obj.spectral_density = system['sd']          # full memory: the influence functional is exact
    return obj


def main(sys_name, system, N_dyn=None, N_res=None):
    # the driver may be called with two positional arguments, as the old per-study scripts were
    N_dyn = int(sys.argv[1]) if N_dyn is None and len(sys.argv) > 1 else (N_dyn or 6)
    N_res = int(sys.argv[2]) if N_res is None and len(sys.argv) > 2 else (N_res or 6)

    for N in range(1, N_dyn + 1):
        obj = build(system, N)
        obj.run()
        obj.dump(data_path(sys_name, 'Propagation', f'dyn_N{N}.json'))

    for N in range(2, N_res + 1):
        for g in range(2, N + 2):
            build(system, N, 'lp', g).dump(
                data_path(sys_name, 'Propagation', f'res_N{N}_lp_g{g}.json'), levels=LEVELS)
        build(system, N, 'walsh_fourier').dump(
            data_path(sys_name, 'Propagation', f'res_N{N}_walsh_g0.json'), levels=LEVELS)

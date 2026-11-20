"""Circuit-side runs for the weakly damped two-level system.

Every measurement here is of the construction itself: how the influence functional is cut, what
the cut costs in gates and shots, and what the circuit returns. The classical path filter is the
other side of the study and lives in Running/OFPF/.

  [Output] Data/TLS/{Propagation,Grouping,Walsh,Readout}/
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.cx_systems_lib import tls, sim2001, ohmic
from measure.CX_coupling_sweep import LAT_TLS
from lib.runner import run_steps

SYSTEM = 'TLS'
SY = tls()


def propagation():
    # the audit dump: one run per N, then the resource sweep over every group size.
    # N_dyn = 5 and N_res = 4 are what Data/TLS/Propagation holds; N = 6 is a 35-qubit statevector
    # (550 GB) and no table reads it, so the sweep stops where the paper stops.
    from measure import CX_propagation
    CX_propagation.main(SYSTEM, SY, N_dyn=5, N_res=4)


def grouping():
    # smallest exact group size g* against N, on the blip lattice of a two-valued s
    from measure import CX_gstar_vs_N
    CX_gstar_vs_N.SYSTEM = SYSTEM
    CX_gstar_vs_N.CASES = ((SY, LAT_TLS, 1, 'TLS'),)
    CX_gstar_vs_N.main()


def walsh():
    # the fused Walsh dilation against the grouped one, exact and at the optimum
    from measure import CX_walsh_vs_lp_exact, CX_walsh_vs_lp_opt
    for mod in (CX_walsh_vs_lp_exact, CX_walsh_vs_lp_opt):
        mod.SYSTEM = SYSTEM
        mod.N_MAX = 6                           # both files carry N = 2..6 in the paper's SI table
        mod.CASES = ((SY, 'TLS'),)
        mod.main()


def readout():
    # full readout against the two truncated ones, with the classical path sum as reference
    from measure import CX_results_readout_data
    CX_results_readout_data.SYSTEM = SYSTEM
    CX_results_readout_data.CASES = ((SY, 8),)
    CX_results_readout_data.main()


def psd():
    # NEEDS cvxpy: the semidefinite condition on the same M, with a bath made Markovian
    from measure import CX_grouping_psd_vs_lattice as psd_mod
    DT, BETA, XI = 0.3, 0.720, 2 * 2.094 / 3.141592653589793
    W_C = 1.0 / (0.25 * DT)                     # 0.25 time slices per bath correlation time
    psd_mod.SYSTEM = SYSTEM
    psd_mod.FILENAME = 'psd_vs_lattice_markov.json'
    psd_mod.CASE = dict(sim2001(BETA), name='Markov-bath on TLS lattice', sd=ohmic(XI, W_C))
    psd_mod.LATTICE = LAT_TLS                   # s = (+1, -1): the three-valued lattice, same M
    psd_mod.N_Q = 1
    psd_mod.main()


STEPS = (('propagation', propagation), ('grouping', grouping), ('walsh', walsh),
         ('readout', readout), ('psd', psd))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

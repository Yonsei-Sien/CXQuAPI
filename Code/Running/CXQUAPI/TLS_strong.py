"""Circuit-side runs for the strongly damped two-level system.

Same measurements as the weakly damped case, on the bath that makes the rescaling bite: this is
where the per-component dilation runs away and where the linear program is worth its cost.

  [Output] Data/TLS_strong/{Propagation,Grouping,LP,Walsh,Validation,Readout}/
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.cx_systems_lib import tls, ohmic
from measure.CX_coupling_sweep import LAT_TLS
from lib.runner import run_steps

SYSTEM = 'TLS_strong'
# the strong bath is the weak system's Hamiltonian under xi = 1.2, w_c = 2.5, beta = 0.2
SY = dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2)


def grouping():
    from measure import CX_gstar_vs_N, CX_grouping_hierarchy
    CX_grouping_hierarchy.main()            # alpha and CNOT of pairs vs g-windows, N = 1..7
    CX_gstar_vs_N.SYSTEM = SYSTEM
    CX_gstar_vs_N.CASES = ((SY, LAT_TLS, 1, 'TLS-strong'),)
    CX_gstar_vs_N.main()


def lp():
    # what a group below g* costs: alpha, its shots, and the gates it saves
    from measure import CX_lp_cost_vs_shots, CX_lp_time_vs_shot_time
    CX_lp_cost_vs_shots.SYSTEM = SYSTEM
    CX_lp_cost_vs_shots.CASES = ((dict(tls(), sd=ohmic(1.2, 2.5), beta=0.2), LAT_TLS, 1, 'TLS-strong'),)
    CX_lp_cost_vs_shots.TOPS = {'TLS-strong': 9}
    CX_lp_cost_vs_shots.main()
    CX_lp_time_vs_shot_time.SYSTEM = SYSTEM
    CX_lp_time_vs_shot_time.SYSTEMS = {'TLS-strong': dict(tls(), sd=ohmic(1.2, 2.5), beta=0.2)}
    CX_lp_time_vs_shot_time.main()


def walsh():
    from measure import CX_walsh_vs_lp_exact, CX_walsh_vs_lp_opt
    for mod in (CX_walsh_vs_lp_exact, CX_walsh_vs_lp_opt):
        mod.SYSTEM = SYSTEM
        mod.N_MAX = 6                           # both files carry N = 2..6 in the paper's SI table
        mod.CASES = ((SY, 'TLS-strong'),)
        mod.main()


def readout():
    from measure import CX_results_readout_data, CX_readout_rules
    CX_results_readout_data.SYSTEM = SYSTEM
    CX_results_readout_data.CASES = ((SY, 8),)
    CX_results_readout_data.main()
    CX_readout_rules.SYSTEM = SYSTEM
    CX_readout_rules.SYSTEMS = (SY,)
    CX_readout_rules.main(N=5)


def psd():
    # NEEDS cvxpy: the semidefinite condition against the finite-lattice one, on this system's own bath
    from measure import CX_grouping_psd_vs_lattice
    CX_grouping_psd_vs_lattice.SYSTEM = SYSTEM
    CX_grouping_psd_vs_lattice.main()


STEPS = (('grouping', grouping), ('lp', lp), ('walsh', walsh),
         ('readout', readout), ('psd', psd))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

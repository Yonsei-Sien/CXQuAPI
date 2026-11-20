"""Circuit-side runs for the three-state model.

d = 3 on two qubits per slice, with the fourth register state decoupled. This is where the
construction is shown to run past two states, and where the readout is read from a superposed
initial state.

  [Output] Data/Sim2001/{Propagation,Grouping,LP,Walsh,Readout}/
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.cx_systems_lib import sim2001, ohmic
from measure.CX_coupling_sweep import LAT_3LS
from lib.runner import run_steps

SYSTEM = 'Sim2001'
SY = sim2001(0.72)


def grouping():
    # g* against N, and what reducing the blip lattice does to it
    from measure import CX_gstar_vs_N, CX_lattice_variants
    CX_gstar_vs_N.SYSTEM = SYSTEM
    CX_gstar_vs_N.CASES = ((SY, LAT_3LS, 2, 'Sim2001'),)
    CX_gstar_vs_N.main()
    CX_lattice_variants.SYSTEM = SYSTEM
    CX_lattice_variants.CASES = ((SY, 2, 'Sim2001'),)
    CX_lattice_variants.main()


def lp():
    from measure import CX_lp_cost_vs_shots, CX_lp_time_vs_shot_time
    CX_lp_cost_vs_shots.SYSTEM = SYSTEM
    CX_lp_cost_vs_shots.CASES = ((SY, LAT_3LS, 2, 'Sim2001'),)
    CX_lp_cost_vs_shots.TOPS = {'Sim2001': 8}
    CX_lp_cost_vs_shots.main()
    CX_lp_time_vs_shot_time.SYSTEM = SYSTEM
    CX_lp_time_vs_shot_time.SYSTEMS = {'Sim2001': SY}
    CX_lp_time_vs_shot_time.main()


def walsh():
    from measure import CX_walsh_vs_lp_exact, CX_walsh_vs_lp_opt
    for mod in (CX_walsh_vs_lp_exact, CX_walsh_vs_lp_opt):
        mod.SYSTEM = SYSTEM
        # d = 3 at N = 6 is 2e8 CNOT and qiskit's high-level synthesis aborts on it
        # (pyo3 PanicException: EdgeIxLimit), which kills the process and the opt pass with it
        mod.N_MAX = 5
        mod.CASES = ((SY, 'Sim2001'),)
        mod.main()


def readout():
    from measure import CX_results_readout_data, CX_readout_rules
    CX_results_readout_data.SYSTEM = SYSTEM
    CX_results_readout_data.CASES = ((SY, 6),)
    CX_results_readout_data.main()
    CX_readout_rules.SYSTEM = SYSTEM
    CX_readout_rules.SYSTEMS = (SY,)
    CX_readout_rules.main(N=5)


def superposed():
    # every population from rho(0) = |psi><psi|, |psi> = (|1> + |3>)/sqrt(2), read in full.
    # V is raised to 1 so that the middle state is populated; the published V leaves it empty.
    from measure import CX_super_pops
    CX_super_pops.SYSTEM = SYSTEM
    # V = 1 is the model's default now, so this is the same system every other step uses
    CX_super_pops.CASES = ((sim2001(0.72), 8, (0, 2)),)
    CX_super_pops.main()


def psd():
    # NEEDS cvxpy: the same comparison with the bath correlation time inside one time step
    from measure import CX_grouping_psd_vs_lattice as psd_mod
    DT, BETA, XI = 0.3, 0.720, 2 * 2.094 / 3.141592653589793
    W_C = 1.0 / (0.25 * DT)
    psd_mod.SYSTEM = SYSTEM
    psd_mod.FILENAME = 'psd_vs_lattice_markov.json'
    psd_mod.CASE = dict(sim2001(BETA), name='Sim2001-markov', sd=ohmic(XI, W_C))
    psd_mod.LATTICE = LAT_3LS       # s = (-1, 0, 1), so Delta runs over five values rather than three
    psd_mod.N_Q = 2                 # two path qubits per slice: a group of g costs 2^(4g)
    psd_mod.main()

    # the same scan on the published bath rather than the Markovian one: tab:psd3's second half
    psd_mod.FILENAME = 'psd_vs_lattice_ownbath.json'
    psd_mod.CASE = SY
    # the table ends at N = 6; on this bath no g reaches alpha = 1 before the fused one, so N = 7
    # would only add a 4.3e9-CNOT group that no circuit could carry
    psd_mod.N_RANGE = range(2, 7)
    psd_mod.main()


STEPS = (('grouping', grouping), ('lp', lp), ('walsh', walsh),
         ('readout', readout), ('superposed', superposed), ('psd', psd))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

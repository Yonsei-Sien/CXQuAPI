"""Circuit-side runs for the eight-level tight-binding chain.

d = 8 on three qubits per slice, carrying the bath, the coupling and the time step of the weakly
damped two-level system, so that d is the only difference between them. Readout only: the chain
is here to show that the construction runs at d = 8, not to say anything new about the bath.

  [Output] Data/Chain8/Readout/
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.cx_systems_lib import chain
from lib.runner import run_steps

SYSTEM = 'Chain8'


def readout():
    from measure import CX_results_readout_data
    CX_results_readout_data.SYSTEM = SYSTEM
    CX_results_readout_data.CASES = ((chain(8), 4),)
    CX_results_readout_data.main()


def superposed():
    # |psi> = (|1> + |5>)/sqrt(2). NOT a state symmetric about the centre of the chain: the chain
    # reverses and the influence functional is invariant under s -> -s, so such a state would force
    # the eight populations into four exactly degenerate pairs.
    from measure import CX_super_pops
    CX_super_pops.SYSTEM = SYSTEM
    CX_super_pops.CASES = ((chain(8), 5, (0, 4)),)
    CX_super_pops.main()


STEPS = (('readout', readout), ('superposed', superposed))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

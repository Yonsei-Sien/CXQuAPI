"""Classical path-filter runs for the strongly damped two-level system.

Nothing here builds a circuit. These measure where the path weight sits -- how far the on-the-fly
filter reaches, and how much of the sum the heaviest paths carry -- which is what decides whether
a truncated readout is worth anything on the circuit side.

  [Output] Data/TLS_strong/{OFPF,Bath}/
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.cx_systems_lib import tls, ohmic
from lib.runner import run_steps

SYSTEM = 'TLS_strong'
SY = dict(tls(), name='TLS-strong', sd=ohmic(1.2, 2.5), beta=0.2)


def reach():
    # how many paths survive the on-the-fly filter, against N and the cutoff
    from measure import CX_ofpf_reach
    CX_ofpf_reach.SYSTEM = SYSTEM
    CX_ofpf_reach.CASES = (SY,)
    CX_ofpf_reach.main()


def concentration():
    # how much of the sum the heaviest paths carry, for populations and then for coherences
    from measure import CX_path_dominance, CX_coherence_dominance
    CX_path_dominance.SYSTEM = SYSTEM
    CX_path_dominance.CASES = ((SY, 8),)
    CX_path_dominance.XI_SCAN = False     # the coupling scan spans systems; Common.py runs it
    CX_path_dominance.main()
    CX_coherence_dominance.SYSTEM = SYSTEM
    CX_coherence_dominance.SYSTEMS = (SY,)
    CX_coherence_dominance.main()


STEPS = (('reach', reach), ('concentration', concentration))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

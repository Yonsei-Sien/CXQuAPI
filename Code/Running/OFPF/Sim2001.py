"""Classical path-filter runs for the three-state model.

This model concentrates for a kinematic reason rather than a dissipative one: its tunnelling is
small against its level spacings, so the free propagator is nearly diagonal and one path carries
the sum whatever the bath does. No coherence table here -- the two-level systems carry that.

  [Output] Data/Sim2001/{OFPF,Bath}/
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.cx_systems_lib import sim2001
from lib.runner import run_steps

SYSTEM = 'Sim2001'
SY = sim2001(0.72)


def reach():
    from measure import CX_ofpf_reach
    CX_ofpf_reach.SYSTEM = SYSTEM
    CX_ofpf_reach.CASES = (SY,)
    CX_ofpf_reach.main()


def concentration():
    from measure import CX_path_dominance
    CX_path_dominance.SYSTEM = SYSTEM
    CX_path_dominance.CASES = ((SY, 6),)
    CX_path_dominance.XI_SCAN = False     # the coupling scan spans systems; Common.py runs it
    CX_path_dominance.main()


STEPS = (('reach', reach), ('concentration', concentration))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

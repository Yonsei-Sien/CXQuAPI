"""Classical path-filter runs that span systems rather than belonging to one.

Two scans read across the whole set: how the concentration moves with the system--bath coupling,
and how it moves with the shape of the bath. These write to the systems the scan itself names --
TLS, 3LS and TLS_shape -- which is why no single system's driver owns them.

  [Output] Data/Common/Bath/path_dominance_xi.json,
           Data/{TLS,3LS,TLS_shape}/Bath/coupling_sweep.json, Data/Common/Bath/ (markovian shape)
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.runner import run_steps

SYSTEM = 'Common'


def concentration():
    # the coupling scan only: no per-system table, the system drivers wrote those
    from measure import CX_path_dominance
    CX_path_dominance.SYSTEM = SYSTEM
    CX_path_dominance.CASES = ()
    CX_path_dominance.XI_SCAN = True
    CX_path_dominance.main()


def coupling():
    # concentration against xi, then against the cutoff that sets the bath correlation time
    from measure import CX_coupling_sweep, CX_markovian_shape
    CX_coupling_sweep.main()
    CX_markovian_shape.SYSTEM = SYSTEM
    CX_markovian_shape.main(N=5)   # positional argv belongs to the coupling sweep above


def fullwindow():
    # lambda_min(M) and the largest full-window generator over the whole (beta, w_c dt) grid
    from measure import CX_markovian_full_window
    CX_markovian_full_window.main()


STEPS = (('concentration', concentration), ('coupling', coupling), ('fullwindow', fullwindow))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

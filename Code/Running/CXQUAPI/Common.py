"""Circuit-side runs that span systems rather than belonging to one.

Four measurements that are read across the whole set at once, so no single system owns them.
The order matters here and nowhere else: `shots` reads the gate counts `cost` wrote.

  [Output] Data/<Sys>/Cost/cost_memory.json, Data/<Sys>/Shots/shot_cost.json,
           Data/Common/Readout/{layer_hadamard_ns0.json, fixed_cutoff_sweep.json},
           Data/Common/Validation/exactness_direct.json
"""
import sys

MASTER = '/home/bjbowling123/Project/CX_QuAPI'
sys.path[:0] = ['/home/bjbowling123/Project/PyQAKE/Code/Package', MASTER + '/Code']

from lib.runner import run_steps

SYSTEM = 'Common'


def cost():
    # gates and ancillas against the retained memory, for every system at once
    from measure import CX_cost_memory_scan
    CX_cost_memory_scan.main()


def shots():
    # RUN AFTER cost: this reads the counts that cost wrote
    from measure import CX_shot_cost_data
    CX_shot_cost_data.main()


def readout():
    # the readout layer itself at exact marginals, then the fixed-cutoff sweep. The sampled variant
    # (ns = 3000) was dropped on 2026-11-18: Table 3 quotes ns = 0 and nothing read the sampled file.
    from measure import CX_layer_hadamard, CX_fixed_cutoff_sweep
    CX_layer_hadamard.SYSTEM = SYSTEM
    CX_layer_hadamard.N_SAMPLE = 0
    CX_layer_hadamard.main()
    CX_fixed_cutoff_sweep.SYSTEM = SYSTEM
    CX_fixed_cutoff_sweep.main()


def dominance():
    # one number, the diagonal dominance of M, against the smallest exact group g*
    from measure import CX_fig3_dominance
    CX_fig3_dominance.main()


def validation():
    # the circuit against the classical path sum, element by element
    from measure import CX_exactness_direct
    CX_exactness_direct.SYSTEM = SYSTEM
    CX_exactness_direct.main()


STEPS = (('cost', cost), ('shots', shots), ('readout', readout), ('dominance', dominance),
         ('validation', validation))

if __name__ == '__main__':
    sys.exit(run_steps(STEPS))

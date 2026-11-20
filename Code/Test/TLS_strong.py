"""One-system test of CX-QuAPI: the strongly damped two-level system, run end to end from pyqake alone.

Nothing outside pyqake.ansatz is imported. The system is written out below, the circuit is built and read by
pyqake.ansatz.CXQUAPI, and the answer is checked against pyqake.ansatz.QUAPI.FPF -- a classical path sum with
its own influence coefficients and no code shared with the circuit -- and against the values the paper prints.

System (spin-boson, the paper's "strongly damped two-level system"):
    H = Omega sigma_x (Omega = 1),   S = sigma_z,   dt = 0.25,   beta = 0.2
    J(w) = (pi/2) hbar xi w exp(-|w|/w_c),   xi = 1.2,   w_c = 2.5,   full memory (N_memory = N)

What is checked, per group size g:
    1. exactness   rho(N dt) from the circuit equals the classical FPF          |diff| < 1e-10, whatever alpha is
    2. program     alpha (LP optimum) equals the value in the paper's Table S2
    3. encoding    prod_G lambda_G, read off the diagonals the circuit encodes, equals alpha
    4. gates       CNOTs of the dilation equal Table S2 (sum_G 2^{2|G|n_q})
    5. readout     P(readout) of each population is rho_aa^2 / (d^{2(N-1)} alpha^2): the dilution of the path
                   register times the rescaling penalty. This is the law Fig. 1b plots
    6. shots       P(readout) at g relative to the smallest group with alpha = 1 equals 1/alpha^2: the rescaling
                   costs exactly alpha^2 in shots, and nothing else in the run depends on g

    python3 TLS_strong.py                 # N = 3, g = 2 and 3 (seconds)
    python3 TLS_strong.py --N 6 --g 5 6   # the pair quoted in the cover letter (lattice g* = 5, PSD needs 6)

  [Input]  --N   : number of time steps
           --g   : group sizes of the LP family (2 = pairs, else windows of g slices)
           --out : directory for the per-run json (default: out/ beside this file)
  [Output] out/TLS_strong_N{N}_g{g}.json : CXQUAPI.dump() -- config, groups, alpha, lambdas, CNOTs, RDM
           stdout : one line per check, PASS/FAIL, and exit status 1 if any check failed
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from pyqake.ansatz.CXQUAPI import CXQUAPI, h_bar
from pyqake.ansatz.QUAPI import FPF


# ---------------------------------------------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------------------------------------------
H_C   = np.array([[0, 1], [1, 0]], dtype=complex)            # Omega sigma_x, Omega = 1
S_C   = np.array([1.0, -1.0])                                  # sigma_z eigenvalues: the positions a path visits
DT    = 0.25
BETA  = 0.2
XI    = 1.2                                                    # Kondo parameter: strong damping
W_C   = 2.5                                                    # bath cutoff frequency


# Ohmic bath, J(w) = (pi/2) hbar xi w exp(-|w|/w_c); the influence coefficients eta_kk' follow from it
def spectral_density(w):
    return 0.5 * np.pi * h_bar * XI * w * np.exp(-abs(w) / W_C)


# The paper's Table S2 (TLS-strong): (N, g) -> (CNOT of the dilation, alpha). Copied from si.tex, tab:psd.
TABLE_S2 = {(2, 2): (48, 2.2150),   (2, 3): (64, 1.0),
            (3, 2): (96, 23.700),   (3, 3): (144, 1.0),    (3, 4): (256, 1.0),
            (4, 3): (240, 2.9154),  (4, 4): (528, 1.0),    (4, 5): (1024, 1.0),
            (5, 4): (816, 2.1153),  (5, 5): (2064, 1.0),
            (6, 4): (1120, 6.4329), (6, 5): (3120, 1.0),   (6, 6): (8208, 1.0),
            (7, 5): (4192, 1.8355), (7, 6): (12336, 1.0),  (7, 7): (32784, 1.0)}

TOL_RDM   = 1e-10                                              # circuit vs classical path sum
TOL_ALPHA = 5e-4                                               # relative: Table S2 prints four or five digits
TOL_SHOTS = 1e-8                                               # relative: two runs of the same circuit family


# ---------------------------------------------------------------------------------------------------------------
# Classical reference
# ---------------------------------------------------------------------------------------------------------------
# rho(N dt) by FPF, full memory. FPF prints instead of returning and saves one {t}.npy per step.
#   [Input]  N : number of time steps
#   [Output] rho : 2 x 2 reduced density matrix, rho(0) = |0><0|
def classical_rho(N):
    class Bath(FPF):
        def spectral_density(self, w):
            return spectral_density(w)

    rho0 = np.zeros((2, 2), dtype=complex)
    rho0[0, 0] = 1.0
    with tempfile.TemporaryDirectory() as chk, contextlib.redirect_stdout(io.StringIO()):
        # t_max + 1e-9: FPF takes int(t_max // dt) steps, and N*dt alone can land one step short
        fpf = Bath(rho0, H_C, S_C, N * DT + 1e-9, DT, N_memory=N, beta=BETA, chkfile=chk)
        fpf.run()
        return np.load(f'{chk}/{np.around(N * DT, 3)}.npy')


# ---------------------------------------------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------------------------------------------
# Build, run and dump one CX-QuAPI circuit with the LP-grouped influence functional.
#   [Input]  N, g : time steps and group size
#            out  : directory of the json record
#   [Output] cx   : the CXQUAPI object after run(), holding alpha_lp, lambdas, cx_breakdown, p_readout
#            rho  : 2 x 2 reduced density matrix, trace-normalized
def quantum_rho(N, g, out):
    to_pauli = lambda M: SparsePauliOp.from_operator(Operator(M), atol=0, rtol=0)   # keep every coefficient
    cx = CXQUAPI(to_pauli(H_C), to_pauli(np.diag(S_C).astype(complex)), DT, N,
                 N_memory=N, N_state=2, beta=BETA, IF='lp', IF_setting={'group_size': g}, verbose=False)
    cx.spectral_density = spectral_density                     # the same bath the classical reference sees
    with contextlib.redirect_stdout(io.StringIO()):
        rho = cx.run()                                       # Aer when installed and worth it, else Statevector
        cx.dump(os.path.join(out, f'TLS_strong_N{N}_g{g}.json'), circuit=False)
    return cx, rho


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--N', type=int, default=3)
    ap.add_argument('--g', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    N, failed = args.N, []

    def check(name, ok, detail):
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<10} {detail}', flush=True)
        if not ok:
            failed.append(name)

    print(f'CX-QuAPI test: strongly damped two-level system, N = {N}, g = {args.g}', flush=True)
    rho_ref = classical_rho(N)
    print(f'  FPF  rho_00 = {rho_ref[0, 0].real:.10f}   rho_01 = {rho_ref[0, 1].real:+.10f}{rho_ref[0, 1].imag:+.10f}i\n',
          flush=True)

    runs = {}
    for g in args.g:
        cx, rho = quantum_rho(N, g, args.out)
        runs[g] = cx
        alpha, prod_lam = float(cx.alpha_lp), float(np.prod(cx.lambdas))
        print(f'g = {g}:  {len(cx.groups)} groups, {cx.N_ancilla} ancillas, alpha = {alpha:.6g}', flush=True)

        # 1. the circuit is exact whatever alpha is: rescaling changes the shot count, not the answer
        diff = float(np.max(np.abs(rho - rho_ref)))
        check('exactness', diff < TOL_RDM, f'max |rho_circuit - rho_FPF| = {diff:.2e}')

        # 2-4. against the paper's table, where it has a row for (N, g)
        ref = TABLE_S2.get((N, g))
        if ref is not None:
            cnot_ref, alpha_ref = ref
            check('program', abs(alpha - alpha_ref) <= TOL_ALPHA * alpha_ref,
                  f'alpha = {alpha:.6g}   (Table S2: {alpha_ref:g})')
            check('gates', cx.cx_breakdown['hermitian'] == cnot_ref,
                  f'dilation CNOT = {cx.cx_breakdown["hermitian"]}   (Table S2: {cnot_ref})')
        else:
            print(f'  [ -- ] program    no Table S2 row for N = {N}, g = {g}')
        # the two numbers come from different objects: the LP's optimum, and the diagonals the circuit encodes
        check('encoding', abs(prod_lam - alpha) <= 1e-9 * alpha,
              f'prod lambda = {prod_lam:.10g}   alpha = {alpha:.10g}')
        # 5. P(readout) = rho_aa^2 / (D alpha^2), D = d^{2(N-1)} intermediate path configurations
        D = 2 ** (2 * (N - 1))
        want = np.real(np.diag(rho_ref)) ** 2 / (D * alpha ** 2)
        err = float(np.max(np.abs(np.array(cx.p_readout) / want - 1)))
        check('readout', err < TOL_SHOTS, f'P(readout) / (rho_aa^2 / D alpha^2) - 1 = {err:.1e}')
        p0 = cx.p_readout[0]                                     # rho_00, the element Fig. 1b plots
        print(f'  P(readout, rho_00) = {p0:.4e}  ->  {(1 - p0) / (1e-4 * p0):.3e} shots for 1% accuracy\n',
              flush=True)

    # 6. the shot penalty is alpha^2 and nothing else: compare every g against a group size that reached alpha = 1
    free = [g for g, cx in runs.items() if cx.alpha_lp <= 1 + 1e-9]
    if free and len(runs) > 1:
        base = runs[min(free)]
        for g, cx in runs.items():
            if g == min(free):
                continue
            ratio = np.array(cx.p_readout) / np.array(base.p_readout)
            want = 1.0 / float(cx.alpha_lp) ** 2
            check('shots', np.allclose(ratio, want, rtol=TOL_SHOTS, atol=0),
                  f'P(readout, g={g}) / P(readout, g={min(free)}) = {ratio.max():.6e}   1/alpha^2 = {want:.6e}')

    print(f'\n{"all checks passed" if not failed else "FAILED: " + ", ".join(failed)}   (json in {args.out})')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

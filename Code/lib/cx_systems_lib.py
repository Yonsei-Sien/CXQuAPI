"""Model systems, classical FPF reference, and cost guards shared by
CX_cost_scaling.py and CX_system_validation.py.

Systems (all with a diagonal position operator S, so every IF mode is exact):
  tls()     : symmetric spin-boson TLS, Makri & Makarov JCP 102, 4600 (1995) Table I
              H = Omega X, S = Z, xi = 0.1, w_c = 7.5, beta = 5, dt = 0.25
  dba3()    : donor-bridge-acceptor, Sim & Makri JPCB 101, 5446 (1997) config C2 + SD1,
              padded to 2 qubits with a decoupled dummy state
  chain(d)  : d-site tight-binding chain, H_ij = -1 (|i-j|=1), S = diag(linspace(1,-1,d)),
              same bath as tls(). chain(2) is H = -X, S = Z (Case1).

  [Output] system dict : name, d, N_state, H_q, S_q (SparsePauliOp), H_c, S_c (arrays),
                         dt, beta, sd (spectral density), s0 (label)
"""
import contextlib
import io
import tempfile

import numpy as np
from scipy.linalg import block_diag
from qiskit.quantum_info import Operator, SparsePauliOp

from pyqake.ansatz.CXQUAPI import CXQUAPI, h_bar
from pyqake.ansatz.QUAPI import FPF

from lib.cxquapi_herm_lib import CXQUAPI_HERM


# SparsePauliOp.from_operator's default rtol silently drops small coefficients -- keep exact
def to_pauli(M):
    return SparsePauliOp.from_operator(Operator(M), atol=0, rtol=0)


def ohmic(xi, w_c):
    def sd(w):
        return 0.5 * np.pi * h_bar * xi * w * np.exp(-abs(w) / w_c)
    return sd


def tls():
    H = np.array([[0, 1], [1, 0]], dtype=complex)                 # Omega X, Omega = 1
    S = np.diag([1.0, -1.0]).astype(complex)                      # sigma_z
    return dict(name='TLS', d=2, N_state=2, H_q=to_pauli(H), S_q=to_pauli(S),
                H_c=H, S_c=np.diag(S), dt=0.25, beta=5.0, sd=ohmic(0.10, 7.5), s0='0')


def dba3():
    E2, E3, V12, V23 = -400.0, -2000.0, 22.0, 135.0              # cm^-1
    H3 = np.array([[0, V12, 0], [V12, E2, V23], [0, V23, E3]], dtype=complex)
    S3 = np.diag([-1.0, 0.0, 1.0]).astype(complex)
    dt = 2 * np.pi * 2.99792458e-2 * 0.005                        # 5 fs in cm^-1 code units
    beta = 1.0 / (0.695034800 * 300.0)                            # 300 K
    return dict(name='DBA3', d=3, N_state=3,
                H_q=to_pauli(block_diag(H3, [[0.0]])), S_q=to_pauli(block_diag(S3, [[0.0]])),   # dummy s = 0: keeps the blip lattice physical
                H_c=H3, S_c=np.diag(S3), dt=dt, beta=beta, sd=ohmic(1.67, 600.0), s0='00')


# Three-state model of E. Sim, J. Chem. Phys. 115, 4450 (2001), Eqs. (21)-(22), reduced units (hbar = 1):
#   H_s = [[0, V, 0], [V, E2, V], [0, V, E3]],  E2 = E3/2 = -1.3333333,  s = (-1, 0, 1)
#   V = 0.066667 AS PUBLISHED. This module's default is V = 1 -- see the block below for why.
#   J(w) = eta w exp(-w/w_c),  eta = 2.094, w_c = 1   ->  ohmic(xi) with (pi/2) xi = eta
#   dt = 0.3, bath memory ~ 4.8 = 16 dt, rho(0) = |1><1|.
# beta is NOT stated there (it follows Sim & Makri, Comput. Phys. Commun. 99, 335 (1997)); pass it explicitly.
# The fourth (dummy) register state gets s = 0, inside the physical range, so paths through it -- which carry
# zero amplitude, H being block diagonal -- never widen the blip lattice or inflate any lambda.
# The three-state model. H_S = [[0, V, 0], [V, E2, V], [0, V, 2 E2]], a chain of three levels with
# equal nearest-neighbour tunnelling V on a ladder of spacing |E2|, coupled to the bath through
# s = (-1, 0, 1).
#
# V = 1 IS THE MODEL (2026-11-13, Sien's decision). The published V = 0.066667 is small against the
# level spacings 20V and 40V, which makes the free propagator nearly diagonal: the middle state then
# stays empty to 2e-3 forever, and one path carries the whole sum whatever the bath does. The paper
# therefore showed a three-state model whose concentration was kinematic, which is not the property
# the paper is about. At V = 1 the propagator is far from diagonal and every state is populated, so
# every three-state result in the paper is now measured on this one model.
#
# What V does and does not change: the blip matrix M is set by the bath, dt and S and not by H_S, so
# g* and alpha are identical at both V (g* = 6, alpha(pairs) = 36.1387 at N = 5). Anything read off
# the path weights carries U = exp(-i H_S dt / hbar) and therefore does move with V -- at N = 6 the
# participation ratio goes 1.02 -> 22.85 and D/R(0.1%) goes 4218 -> 44.5.
#
# The name carries V only when it is NOT the model's V, so files and registry keys written for the
# model keep the plain spelling; sim2001(beta, V=0.066667) still reproduces the published variant and
# labels itself as such.
def sim2001(beta, V=1.0):
    E2 = -1.3333333
    H3 = np.array([[0, V, 0], [V, E2, V], [0, V, 2 * E2]], dtype=complex)
    S3 = np.diag([-1.0, 0.0, 1.0]).astype(complex)
    tag = f'Sim2001(beta={beta:g})' if V == 1.0 else f'Sim2001(beta={beta:g}, V={V:g})'
    return dict(name=tag, d=3, N_state=3,
                H_q=to_pauli(block_diag(H3, [[0.0]])), S_q=to_pauli(block_diag(S3, [[0.0]])),
                H_c=H3, S_c=np.diag(S3), dt=0.3, beta=beta, sd=ohmic(2 * 2.094 / np.pi, 1.0), s0='00')


def chain(d):
    assert d & (d - 1) == 0, 'chain(d) needs d = 2^n (no dummy padding)'
    H = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    S = np.diag(np.linspace(1.0, -1.0, d)).astype(complex)
    nq = int(np.log2(d))
    return dict(name=f'Chain{d}', d=d, N_state=d, H_q=to_pauli(H), S_q=to_pauli(S),
                H_c=H, S_c=np.diag(S), dt=0.25, beta=5.0, sd=ohmic(0.10, 7.5), s0='0' * nq)


# Build a CX-QuAPI object for any of the three IF modes, full memory (N_memory = N)
def build(system, N, IF):
    kw = dict(N_memory=N, N_state=system['N_state'], beta=system['beta'])
    if IF == 'sz_nagy_herm':
        obj = CXQUAPI_HERM(system['H_q'], system['S_q'], system['dt'], N, **kw)
    else:
        obj = CXQUAPI(system['H_q'], system['S_q'], system['dt'], N, IF=IF, **kw)
    obj.spectral_density = system['sd']
    return obj


# Classical reference rho(N dt). QUAPI.FPF returns nothing -- it writes {t}.npy per step.
#   [Input]  N_memory : memory span, default N (full memory, exact IF)
#   [Output] RDM : N_state x N_state
def run_fpf(system, N, N_memory=None):
    sd = system['sd']

    class _FPF(FPF):
        def spectral_density(self, w):
            return sd(w)

    RDM0 = np.zeros((system['N_state'],) * 2, dtype=complex)
    RDM0[0, 0] = 1.0
    with tempfile.TemporaryDirectory() as chk, contextlib.redirect_stdout(io.StringIO()):
        f = _FPF(RDM0, system['H_c'], system['S_c'], N * system['dt'] + 1e-9, system['dt'],
                 N_memory=N if N_memory is None else N_memory, beta=system['beta'])
        f.chkfile = chk
        f.run()
        return np.load(f'{chk}/{np.around(N * system["dt"], 3)}.npy')


# ---- cost guards (predict BEFORE building anything dense) ----
QSD_CX = {3: 19, 5: 423, 7: 7319, 9: 119383}                 # measured on k00, qs_decomposition

def qsd_cx(n):
    return QSD_CX.get(n, int(round(23 / 48 * 4 ** n)))


def n_terms_full_memory(N):
    return (N + 1) * (N + 2) // 2                            # (k, k') with 0 <= k' <= k <= N


def fpf_paths(d, N):
    return d ** (2 * (N + 1))


def sim_cost(n_ops, n_qubits, n_columns):
    return n_ops * (2 ** n_qubits) * n_columns                # Statevector.evolve ~ ops x 2^Q per column

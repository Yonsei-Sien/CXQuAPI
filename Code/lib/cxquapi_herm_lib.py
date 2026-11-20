"""Sz.-Nagy dilation restricted to the HERMITIAN part of the influence-functional exponent.

For a diagonal position operator S (QuAPI/DVR convention) every IF term exponent d is a
diagonal operator, so its Cartesian split is exact:

    exp(d) = exp(i Im d) * exp(Re d)          ([Re d, i Im d] = 0 for diagonal d)

  * exp(i Im d)  : unit modulus, a diagonal UNITARY  -> no ancilla, no post-selection.
                   All terms are fused into ONE Pauli-Z rotation network, O(N L) 2-body strings.
  * exp(Re d)    : real, positive diagonal -> the only non-unitary factor, dilated per term
                   with one ancilla as a uniformly-controlled Ry:
                       Ry(theta_x)|0> = cos(theta_x/2)|0> + sin(theta_x/2)|1>,
                       cos(theta_x/2) = sigma_x = exp(Re d(x)) / lambda
                   so the ancilla-|0> block is exactly diag(sigma). Cost: 2^n CNOT on n controls.

Compared with 'sz_nagy' (dense Sz.-Nagy unitary of exp(d) on n+1 qubits, synthesized by
qs_decomposition, ~ (23/48) 4^(n+1) CNOT), the post-selection probability is IDENTICAL:
lambda = max exp(Re d) = max |exp(d)| is the same rescaling, and the ancilla-|0> amplitude is
|sigma| in both. The gain is entirely in circuit size.

  [Output] CXQUAPI_HERM : CXQUAPI subclass with IF mode 'sz_nagy_herm'
  [Output] check_ucry_block : self-check of the UCRY qubit ordering (run before long jobs)
"""
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UCRYGate
from qiskit.quantum_info import Operator, SparsePauliOp

from pyqake.ansatz.CXQUAPI import CXQUAPI, embed_local

COUNT_BASIS = ['cx', 'u3']


# Diagonal of a Z/I-only SparsePauliOp: d(x) = sum_j c_j (-1)^popcount(x & mask_j), bit i = qubit i
#   [Input]  op  : SparsePauliOp with no X/Y support
#   [Output] out : complex vector of length 2^num_qubits
def z_diagonal(op):
    n     = op.num_qubits
    basis = np.arange(2 ** n)
    out   = np.zeros(2 ** n, dtype=complex)
    for row, coeff in zip(op.paulis.z, op.coeffs):
        mask = int(np.dot(row.astype(np.int64), 1 << np.arange(n)))
        if hasattr(np, 'bitwise_count'):
            parity = (np.bitwise_count((basis & mask).astype(np.uint64)) & 1).astype(np.int64)
        else:
            parity = np.array([bin(int(x) & mask).count('1') & 1 for x in basis])
        out += coeff * (1 - 2 * parity)
    return out


# Uniformly-controlled Ry block encoding of a real diagonal 0 <= sigma <= 1:
#   <0|_anc U |0>_anc = diag(sigma) on the controls
#   [Input]  sigma : length-2^n real vector
#   [Output] (gate, n) : UCRYGate to be appended on [ancilla] + controls (controls[0] = LSB)
def ucry_encoding(sigma):
    theta = 2 * np.arccos(np.clip(sigma, 0.0, 1.0))
    return UCRYGate([float(t) for t in theta])


# Verifies on a small random instance that the ancilla-|0> block of the UCRY circuit is diag(sigma)
# with the SAME index convention CXQUAPI uses (bit i of the diagonal index = qarg i).
def check_ucry_block(n=3, seed=7):
    rng   = np.random.default_rng(seed)
    sigma = rng.uniform(0.05, 1.0, 2 ** n)
    qc    = QuantumCircuit(n + 1)
    qc.append(ucry_encoding(sigma), [n] + list(range(n)))
    U     = Operator(qc).data
    block = U[:2 ** n, :2 ** n]                 # ancilla = qubit n = highest bit -> first 2^n indices
    err   = np.abs(block - np.diag(sigma)).max()
    assert err < 1e-10, f'UCRY block encoding mismatch: {err:.3e}'
    return err


class CXQUAPI_HERM(CXQUAPI):
    def __init__(self, H, S, dt, N_max_step, N_memory=0, N_state=2, beta=1, IF_setting=None):
        super().__init__(H, S, dt, N_max_step, N_memory=N_memory, N_state=N_state, beta=beta,
                         IF='sz_nagy', IF_setting=IF_setting)
        # Saving Values
        self.IF           = 'sz_nagy_herm'   # Hermitian-part-only per-term dilation
        # Auto-Loaded Values
        self.lambdas      = None             # Per-term rescaling lambda_k = max exp(Re d_k) (>= 1)
        self.cx_breakdown = None             # CNOT count of {'hermitian': UCRY part, 'unitary': Im part}

    # exp(IF) = exp(i Im d_tot) * prod_k exp(Re d_k): fused diagonal unitary + one UCRY ancilla per term
    #   [Output] circuit : transpiled (cx, u3) circuit on 2*N_forward + N_terms qubits
    def gen_IF_circuit(self):
        time_start = time.time()
        IF_coefs   = self.gen_IF_coefs()
        N_qubit    = 2 * self.N_forward
        full_qargs = list(range(N_qubit))
        N_ancilla  = len(IF_coefs)

        herm_circ = QuantumCircuit(N_qubit + N_ancilla)
        im_total  = SparsePauliOp.from_list([('I' * N_qubit, 0.0)])
        lambdas   = []

        for anc, ((k_aft, k_bef), (IF_coef, qargs)) in enumerate(IF_coefs.items()):
            assert not np.any(IF_coef.paulis.x), \
                f"IF term ({k_aft},{k_bef}) has X/Y support -- the Cartesian split is exact only for diagonal S"

            # d = Re d + i Im d, read coefficient-wise since every Z-string is real (+-1)
            re_coef = SparsePauliOp(IF_coef.paulis, IF_coef.coeffs.real)
            im_coef = SparsePauliOp(IF_coef.paulis, IF_coef.coeffs.imag)
            im_total = im_total + embed_local(im_coef, qargs, full_qargs)

            # sigma = exp(Re d) / lambda, lambda = max exp(Re d) -- same rescaling as dilation()
            sigma = np.exp(z_diagonal(re_coef).real)
            lam   = float(sigma.max()) if sigma.max() > 1 else 1.0
            lambdas.append(lam)
            herm_circ.append(ucry_encoding(sigma / lam), [N_qubit + anc] + qargs)

        # exp(i Im d_tot): PauliEvolutionGate synthesizes exp(-i t op), so op = -Im d_tot
        im_total = im_total.simplify()
        im_op    = SparsePauliOp(im_total.paulis, -im_total.coeffs.real)
        unit_circ = QuantumCircuit(N_qubit + N_ancilla)
        unit_circ.append(PauliEvolutionGate(im_op, time=1.0), range(N_qubit))

        herm_circ = transpile(herm_circ, basis_gates=COUNT_BASIS, optimization_level=0)
        unit_circ = transpile(unit_circ, basis_gates=COUNT_BASIS, optimization_level=0)
        circuit   = unit_circ.compose(herm_circ)

        self.N_ancilla    = N_ancilla
        self.lambdas      = lambdas
        self.cx_breakdown = {'hermitian': herm_circ.count_ops().get('cx', 0),
                             'unitary':   unit_circ.count_ops().get('cx', 0)}
        print(f'Hermitian-split Sz.-Nagy IF Circuit Generated ({len(IF_coefs)} terms, {N_ancilla} ancilla, '
              f'CX herm/unit = {self.cx_breakdown["hermitian"]}/{self.cx_breakdown["unitary"]}, '
              f'{time.time() - time_start:.3f}s)')
        return circuit

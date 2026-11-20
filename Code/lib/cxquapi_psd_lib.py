"""PSD term grouping + UCRY encoding: fused-walsh amplitude with per-term (polynomial) gate count.

Per-term dilation loses amplitude that physics does not require. A single coupling term e^{d_kk'} is not a
contraction -- on the path where the coherence flips sign between k and k' it AMPLIFIES (|sigma| > 1) -- so the
block encoding must rescale it by lambda_kk' > 1, and those rescalings multiply: prod_k lambda_k^-2.
The fused influence functional has no such factor (max|sigma| = 1 exactly), but its circuit is exponential in N.

Writing the modulus as a quadratic form in the blip variables Delta_k = s+_k - s-_k,

    Re d^tot = -(1/hbar) Delta^T M Delta,    M_kk = Re eta_kk,   M_kk' = (1/2) Re eta_kk'   (k > k')

a group G has lambda_G = 1 exactly when its own M^(G) is positive semidefinite. So grouping = splitting M into
PSD summands. Pairing coupling (k,k') with a SHARE of both endpoint self-terms,

    M^(kk') = [[w_kk' M_kk, M_kk'], [M_kk', w_k'k M_k'k']],    w_kk' = |M_kk'| / M_kk

makes the determinant vanish exactly (PSD), and the shares fit the budget sum_k' w_kk' <= 1 iff M is diagonally
dominant. Leftover diagonal weight is folded back into one group of that time, so the group generators still sum
to the exact total exponent -- this is an exact regrouping, not an approximation.

Each group's real part is a real diagonal in (0, 1], encoded with one uniformly-controlled Ry on the group's
4 N_qb_state path qubits (2^(4 n_q) CNOT); the imaginary parts of ALL terms are fused into a single diagonal
unitary with no ancilla. Ancillas = number of coupling terms (self terms cost none).

  [Output] CXQUAPI_PSD : CXQUAPI subclass with IF mode 'psd_ucry'
           .lambdas          : per-group rescaling (must be 1.0 when M is diagonally dominant)
           .diag_dominance   : per-time slack M_kk - sum |M_kk'| (negative = not dominant)
           .cx_breakdown     : CNOT of {'hermitian': UCRY part, 'unitary': Im part}
"""
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from pyqake.ansatz.CXQUAPI import alpha_generator, embed_local
from lib.cxquapi_herm_lib import CXQUAPI_HERM, ucry_encoding, z_diagonal, COUNT_BASIS


class CXQUAPI_PSD(CXQUAPI_HERM):
    def __init__(self, H, S, dt, N_max_step, N_memory=0, N_state=2, beta=1, IF_setting=None):
        super().__init__(H, S, dt, N_max_step, N_memory=N_memory, N_state=N_state, beta=beta,
                         IF_setting=IF_setting)
        # Saving Values
        self.IF             = 'psd_ucry'
        # Auto-Loaded Values
        self.diag_dominance = None    # Per-time slack of the weight budget; negative means the closed form fails
        self.weights        = None    # {(k, k'): w_kk'} share of time k's self-term given to coupling (k, k')

    # M_kk = Re eta_kk, M_kk' = (1/2) Re eta_kk' over the RETAINED (memory-truncated) terms only
    def gen_M_matrix(self):
        alphas = alpha_generator(self.dt, self.N_memory, self.N_max_step, self.beta, self.spectral_density)
        M = np.zeros((self.N_max_step + 1, self.N_max_step + 1))
        for k_aft in range(self.N_max_step + 1):
            k_min = 0 if k_aft < self.N_memory else k_aft - self.N_memory
            for k_bef in range(k_min, k_aft + 1):
                value = float(np.real(alphas[k_aft, k_bef]))
                if k_aft == k_bef:
                    M[k_aft, k_aft] = value
                else:
                    M[k_aft, k_bef] = M[k_bef, k_aft] = 0.5 * value
        return M

    # w_kk' = |M_kk'| / M_kk, capped so that sum_k' w_kk' <= 1 (the cap only triggers without diagonal dominance)
    #   [Output] weights : {(k, k'): share of time k's self-term}, leftover : {k: unassigned share}
    def gen_weights(self, M, couplings):
        weights, budget = {}, {k: 0.0 for k in range(self.N_max_step + 1)}
        for (k_aft, k_bef) in couplings:
            for owner, other in ((k_aft, k_bef), (k_bef, k_aft)):
                weights[(owner, other)] = abs(M[k_aft, k_bef]) / M[owner, owner] if M[owner, owner] > 0 else 0.0
                budget[owner] += weights[(owner, other)]
        for (owner, other), value in list(weights.items()):
            if budget[owner] > 1.0:                       # not diagonally dominant: share out what exists
                weights[(owner, other)] = value / budget[owner]
        leftover = {k: max(0.0, 1.0 - min(budget[k], 1.0)) for k in budget}
        return weights, leftover

    # exp(IF) = exp(i Im d_tot) * prod_G exp(Re d_G): one fused diagonal unitary + one UCRY ancilla per COUPLING
    def gen_IF_circuit(self):
        time_start = time.time()
        IF_coefs   = self.gen_IF_coefs()
        N_qubit    = 2 * self.N_forward
        full_qargs = list(range(N_qubit))

        re_part = {kk: SparsePauliOp(c.paulis, c.coeffs.real) for kk, (c, _) in IF_coefs.items()}
        im_part = {kk: SparsePauliOp(c.paulis, c.coeffs.imag) for kk, (c, _) in IF_coefs.items()}
        qargs   = {kk: q for kk, (_, q) in IF_coefs.items()}
        couplings = [kk for kk in IF_coefs if kk[0] != kk[1]]

        M = self.gen_M_matrix()
        weights, leftover = self.gen_weights(M, couplings)
        self.weights        = weights
        self.diag_dominance = [M[k, k] - sum(abs(M[k, j]) for j in range(self.N_max_step + 1) if j != k)
                               for k in range(self.N_max_step + 1)]

        # leftover diagonal weight goes to the first group that contains that time, so the shares still sum to 1
        # (adding a positive diagonal to a PSD 2x2 keeps it PSD, so this cannot break lambda_G = 1)
        extra = {k: 0.0 for k in leftover}
        for k, value in leftover.items():
            for (k_aft, k_bef) in couplings:
                if k in (k_aft, k_bef):
                    extra[k] = value
                    break

        herm_circ = QuantumCircuit(N_qubit + len(couplings))
        lambdas   = []
        used_extra = set()
        for anc, (k_aft, k_bef) in enumerate(couplings):
            group_qargs = qargs[(k_aft, k_bef)]                     # 4 N_qb_state qubits: blocks k_aft and k_bef
            generator   = embed_local(re_part[(k_aft, k_bef)], group_qargs, group_qargs)
            for owner, other in ((k_aft, k_bef), (k_bef, k_aft)):
                share = weights[(owner, other)]
                if owner not in used_extra:                          # give this time's leftover to its first group
                    share += extra[owner]
                    used_extra.add(owner)
                if share:
                    generator = generator + share * embed_local(re_part[(owner, owner)],
                                                                qargs[(owner, owner)], group_qargs)

            sigma = np.exp(z_diagonal(SparsePauliOp.simplify(generator)).real)
            lam   = float(sigma.max()) if sigma.max() > 1 else 1.0  # 1.0 whenever the group is PSD
            lambdas.append(lam)
            herm_circ.append(ucry_encoding(sigma / lam), [N_qubit + anc] + group_qargs)

        # self-terms whose time never appears in a coupling (e.g. N_max_step = 0) still need their real part
        orphan = SparsePauliOp.from_list([('I' * N_qubit, 0.0)])
        for k in range(self.N_max_step + 1):
            if k not in used_extra and (k, k) in re_part:
                orphan = orphan + embed_local(re_part[(k, k)], qargs[(k, k)], full_qargs)
        orphan = SparsePauliOp.simplify(orphan)
        if np.abs(orphan.coeffs).max() > 1e-14:
            sigma = np.exp(z_diagonal(orphan).real)
            lam   = float(sigma.max()) if sigma.max() > 1 else 1.0
            lambdas.append(lam)
            herm_circ = QuantumCircuit(N_qubit + len(couplings) + 1).compose(herm_circ)
            herm_circ.append(ucry_encoding(sigma / lam), [N_qubit + len(couplings)] + full_qargs)

        # exp(i Im d_tot): PauliEvolutionGate synthesizes exp(-i t op), so op = -Im d_tot
        im_total = SparsePauliOp.from_list([('I' * N_qubit, 0.0)])
        for kk in IF_coefs:
            im_total = im_total + embed_local(im_part[kk], qargs[kk], full_qargs)
        im_total  = im_total.simplify()
        unit_circ = QuantumCircuit(herm_circ.num_qubits)
        unit_circ.append(PauliEvolutionGate(SparsePauliOp(im_total.paulis, -im_total.coeffs.real), time=1.0),
                         range(N_qubit))

        herm_circ = transpile(herm_circ, basis_gates=COUNT_BASIS, optimization_level=0)
        unit_circ = transpile(unit_circ, basis_gates=COUNT_BASIS, optimization_level=0)
        circuit   = unit_circ.compose(herm_circ)

        self.N_ancilla    = herm_circ.num_qubits - N_qubit
        self.lambdas      = lambdas
        self.cx_breakdown = {'hermitian': herm_circ.count_ops().get('cx', 0),
                             'unitary':   unit_circ.count_ops().get('cx', 0)}
        print(f'PSD-grouped UCRY IF Circuit Generated ({len(IF_coefs)} terms -> {len(couplings)} groups, '
              f'{self.N_ancilla} ancilla, max lambda_G = {max(lambdas) if lambdas else 1.0:.8f}, '
              f'diag-dominance min slack = {min(self.diag_dominance):.3e}, '
              f'CX herm/unit = {self.cx_breakdown["hermitian"]}/{self.cx_breakdown["unitary"]}, '
              f'{time.time() - time_start:.3f}s)')
        return circuit

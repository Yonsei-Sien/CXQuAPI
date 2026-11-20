"""CX-QuAPI: the quasi-adiabatic propagator path integral as a quantum circuit.

Layout of this file:
  1. Influence coefficients   alpha_generator -- eta_kk' from the spectral density
  2. Grouping program         lp_decomposer -- how to cut the influence functional so that no amplitude is lost
  3. Pieces each construction needs, one section per construction
  4. Readout and path sampling
  5. class CXQUAPI            gen_IF_* -> gen_propagation_circuit -> run

The chunked variant, which propagates a sliding window instead of the whole path register, is in CXCHUNK.py.
"""
import itertools
import json
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation, UnitaryGate, UCRYGate, DiagonalGate
from qiskit.quantum_info import SparsePauliOp, Statevector, PauliList, Operator
from qiskit.synthesis import qs_decomposition
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import linprog


##########################################################################<<< Parameters

k_B         = 1
h_bar       = 1
BASIS_GATES = ['u1', 'u2', 'u3', 'cx', 'id', 'x', 'y', 'z', 'h', 'rz', 'sx', 'cp', 'swap']
COUNT_BASIS = ['cx', 'u3']                                        # basis for CNOT / depth counts
IF_Methods  = {'lp'            : 'LP-grouped uniformly controlled Ry',
               'fused'         : 'Fused IF, Gray-code synthesis',
               'sz_nagy'       : 'Sz.-Nagy dilation per term',
               'walsh_fourier' : 'Fused IF, one parity ladder per Walsh string'}

# ==============================================================================================================
# 1. Influence coefficients
#   eta_kk' from the spectral density; shared by every construction below
# ==============================================================================================================


# QuAPI influence coefficients eta[k_aft, k_bef] (Makri & Makarov), memory-truncated at N_memory
def alpha_generator(dt, N_memory, N_max_step, beta, spectral_density):
    alphas = np.zeros((N_max_step + 1, N_max_step + 1), dtype=complex)
    t_max  = dt * N_max_step

    def integrand(w, k_bef, k_aft):
        J_w     = spectral_density(w)
        thermal = 2 / (1 - np.exp(np.clip(-h_bar * w * beta, -700, 700)))
        shape   = 1
        if 0 < k_bef < k_aft < N_max_step:                                      # interior pair
            shape *= (2 / np.pi) * np.sin(w * dt / 2) ** 2 * np.exp(-1j * w * dt * (k_aft - k_bef))
        if k_aft == k_bef and 0 < k_bef < N_max_step:                            # interior self term
            shape *= (0.5 / np.pi) * (1 - np.exp(-1j * w * dt))
        if k_bef == 0 and k_aft == N_max_step:                                   # both end points
            shape *= (2 / np.pi) * np.sin(w * dt / 4) ** 2 * np.exp(-1j * w * (t_max - dt / 2))
        if k_bef == k_aft and k_aft in (0, N_max_step):                          # end-point self term
            shape *= (0.5 / np.pi) * (1 - np.exp(-1j * w * dt / 2))
        if k_bef == 0 and 0 < k_aft < N_max_step:                                # start point - interior
            shape *= (2 / np.pi) * np.sin(w * dt / 4) * np.sin(w * dt / 2) * np.exp(-1j * w * (k_aft * dt - dt / 4))
        if k_aft == N_max_step and 0 < k_bef < N_max_step:                       # interior - end point
            shape *= (2 / np.pi) * np.sin(w * dt / 4) * np.sin(w * dt / 2) * np.exp(-1j * w * (t_max - k_bef * dt - dt / 4))
        return J_w * thermal * shape / w ** 2

    def integrate(k_bef, k_aft):
        re, _ = quad(lambda w: np.real(integrand(w, k_bef, k_aft)), -np.inf, np.inf)
        im, _ = quad(lambda w: np.imag(integrand(w, k_bef, k_aft)), -np.inf, np.inf)
        return re + 1j * im

    for k_aft in range(N_max_step + 1):
        for k_bef in range(max(0, k_aft - N_memory) if k_aft >= N_memory else 0, k_aft + 1):
            alphas[k_aft, k_bef] = integrate(k_bef, k_aft)
    return alphas


# Print the lower triangle of the influence coefficients
def alpha_printer(alphas):
    print('=====================< IF Coefficient >=====================')
    for i in range(alphas.shape[0]):
        for j in range(i + 1):
            print(f' {i}_{j} : {alphas[i, j]:5f}')

# ==============================================================================================================
# 2. Shared circuit pieces
#   used by more than one construction
# ==============================================================================================================


##########################################################################<<< Quantum Modules

# n-qubit identity
def gen_identity(n):
    return SparsePauliOp.from_list([('I' * n, 1)])


# Position operators placed on (fwd k, bwd k) and (fwd k, bwd k, fwd k', bwd k') local registers
def gen_expanded_position(N_qb_state, position):
    id_1, id_2, id_3 = gen_identity(N_qb_state), gen_identity(2 * N_qb_state), gen_identity(3 * N_qb_state)
    return [[position.expand(id_1), id_1.expand(position)],
            [position.expand(id_3), id_1.expand(position.expand(id_2)),
             id_2.expand(position.expand(id_1)), id_3.expand(position)]]


# Embed a local SparsePauliOp on sub_qargs into full_qargs (identity elsewhere)
def embed_local(op, sub_qargs, full_qargs):
    N_full = len(full_qargs)
    pos    = {q: j for j, q in enumerate(full_qargs)}
    terms  = []
    for pauli, coeff in zip(op.paulis, op.coeffs):
        label = str(pauli)                                                     # label[-1] = local qubit 0
        chars = ['I'] * N_full
        for i, q in enumerate(sub_qargs):
            chars[N_full - 1 - pos[q]] = label[len(sub_qargs) - 1 - i]
        terms.append((''.join(chars), coeff))
    return SparsePauliOp.from_list(terms).simplify()


# Diagonal of a Z-only SparsePauliOp (bit j of the index = qubit j)
def z_diagonal(op):
    basis = np.arange(2 ** op.num_qubits)
    diag  = np.zeros(len(basis), dtype=complex)
    for z_row, coeff in zip(op.paulis.z, op.coeffs):
        mask   = int(np.dot(z_row.astype(np.int64), 1 << np.arange(op.num_qubits)))
        parity = (np.bitwise_count((basis & mask).astype(np.uint64)) & 1).astype(np.int64)
        diag  += coeff * (1 - 2 * parity)
    return diag

# ==============================================================================================================
# 3a. The LP construction  (IF='lp')
#   the method of the paper: cut M by linear program, one UCRY per piece
# ==============================================================================================================


# Blip matrix M of Re IF = -(1/hbar) Delta^T M Delta: M_kk = Re eta_kk, M_kk' = Re eta_kk' / 2
def blip_matrix(alphas, N_memory):
    n_time = alphas.shape[0]
    M = np.zeros((n_time, n_time))
    for k_aft in range(n_time):
        for k_bef in range(max(0, k_aft - N_memory), k_aft + 1):
            value = float(np.real(alphas[k_aft, k_bef]))
            if k_aft == k_bef:
                M[k_aft, k_aft] = value
            else:
                M[k_aft, k_bef] = M[k_bef, k_aft] = 0.5 * value
    return M


# Blip lattice {s_i - s_j} over the physical states
def blip_lattice(s_values):
    return np.unique(np.round(np.subtract.outer(s_values, s_values).ravel(), 12))


# Lattice vectors of length g, one per +-pair, zero excluded
def lattice_vectors(lattice, g):
    seen, vectors = set(), []
    for v in itertools.product(lattice, repeat=g):
        if not any(v) or tuple(-x for x in v) in seen:
            continue
        seen.add(tuple(v)); vectors.append(np.asarray(v, dtype=float))
    return vectors


# All pairs of times (g = 2 family)
def pair_groups(n_time):
    return [(k_bef, k_aft) for k_aft in range(n_time) for k_bef in range(k_aft)]


# Windows of g consecutive times; couplings a window misses are kept as pairs (far_pairs=True)
def window_groups(n_time, g, far_pairs=True):
    if g >= n_time:
        return [tuple(range(n_time))]
    windows = [tuple(range(i, i + g)) for i in range(n_time - g + 1)]
    far     = [(k_bef, k_aft) for k_aft in range(n_time) for k_bef in range(k_aft) if k_aft - k_bef >= g]
    return windows + (far if far_pairs else [])


# Minimal alpha = prod_G lambda_G over splits M = sum_G X_G (exact on the blip lattice; linear program)
#   [Input]  M       : blip matrix, M_kk = Re eta_kk and M_kk' = Re eta_kk'/2
#            groups  : the family of slice tuples the split is allowed to use; every non-zero entry of M must
#                      be covered by at least one group, or the program raises
#            lattice : the values a blip Delta_k = s_i - s_j can take, so the inequalities are imposed only
#                      where the register can actually be evaluated. This is what makes the program linear
#                      rather than semidefinite
#   [Output] alpha    : exp(sum_G t_G / h_bar), the factor the dilation loses in amplitude and therefore the
#                       factor alpha^2 it costs in shots; alpha = 1 means nothing is lost. It is the optimum
#                       over the given family, not over all families, and cannot fall below 1 because every
#                       slack is bounded below by zero
#            X_groups : {group: X_G}, the piece each group carries, symmetric and of size |G| x |G|, with
#                       sum_G X_G = M exactly (an equality constraint, so the residual is solver round-off)
#            t_groups : {group: t_G}, the slack of each group, t_G = max(0, -min_Delta Delta^T X_G Delta).
#                       t_G = 0 says the piece is already a contraction on the lattice and costs nothing;
#                       t_G > 0 is the amplitude that group alone gives up
def optimal_grouping(M, groups, lattice):
    n_time, col, n_var = len(M), [], 0
    for G in groups:
        pairs = [(a, b) for a in range(len(G)) for b in range(a, len(G))]
        col.append({p: n_var + i for i, p in enumerate(pairs)}); n_var += len(pairs)
    t_col = [n_var + i for i in range(len(groups))]; n_var += len(groups)

    eq_rows, eq_rhs = [], []                                                   # sum_G X_G = M
    for i in range(n_time):
        for j in range(i, n_time):
            row = np.zeros(n_var)
            for G, idx in zip(groups, col):
                if i in G and j in G:
                    row[idx[tuple(sorted((G.index(i), G.index(j))))]] = 1.0
            if row.any():
                eq_rows.append(row); eq_rhs.append(M[i, j])
            elif abs(M[i, j]) > 1e-14:
                raise ValueError(f'entry ({i},{j}) of M is covered by no group')

    ub_rows = []                                                               # t_G >= -Delta^T X_G Delta
    for G, idx, t in zip(groups, col, t_col):
        for v in lattice_vectors(lattice, len(G)):
            row = np.zeros(n_var)
            for (a, b), c in idx.items():
                row[c] = -(v[a] * v[a] if a == b else 2.0 * v[a] * v[b])
            row[t] = -1.0
            ub_rows.append(row)

    cost = np.zeros(n_var); cost[t_col] = 1.0
    bounds = [(None, None)] * (n_var - len(groups)) + [(0.0, None)] * len(groups)
    for G, idx in zip(groups, col):                                            # couplings absent from the IF stay zero
        for (a, b), c in idx.items():
            if a != b and abs(M[G[a], G[b]]) < 1e-300:
                bounds[c] = (0.0, 0.0)
    res = linprog(cost, A_ub=np.array(ub_rows), b_ub=np.zeros(len(ub_rows)),
                  A_eq=np.array(eq_rows), b_eq=np.array(eq_rhs), bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(res.message)

    X_groups, t_groups = {}, {}
    for G, idx, t in zip(groups, col, t_col):
        X = np.zeros((len(G), len(G)))
        for (a, b), c in idx.items():
            X[a, b] = X[b, a] = res.x[c]
        X_groups[G], t_groups[G] = X, res.x[t]
    return float(np.exp(sum(t_groups.values()) / h_bar)), X_groups, t_groups

# One call from a blip matrix to the split the circuit will apply.
#   alpha = prod_G lambda_G is minimal over the chosen family; alpha = 1 means no amplitude is lost to the dilation.
#   Pairs whose coupling vanishes are dropped when both their slices are already covered, so that memory-truncated
#   problems do not carry empty groups.
#   [Input]  M       : blip matrix, M_kk = Re eta_kk and M_kk' = Re eta_kk'/2
#            lattice : the values a blip can take, {s_i - s_j}
#            g       : slices per group; 2 gives every pair, larger gives sliding windows plus the far pairs
#   [Output] alpha  : the rescaling the chosen family forces, as in optimal_grouping; the circuit divides
#                     every encoded diagonal by it, and alpha = 1 is the only value that costs no shots
#            X      : {group: X_G}, the split the circuit encodes, one uniformly controlled rotation per group
#            family : the groups in the order the ancillas are assigned, so family[i] is the group ancilla i
#                     carries; len(family) is therefore both the ancilla count and the number of rotations
def lp_decomposer(M, lattice, g):
    n_time = len(M)
    family = pair_groups(n_time) if (g == 2 and n_time > 1) else window_groups(n_time, g)
    family = [G for G in family if not (len(G) == 2 and abs(M[G[0], G[1]]) < 1e-300 and
                                        all(any(t in H for H in family if H != G) for t in G))]
    alpha, X_groups, _ = optimal_grouping(M, family, lattice)
    return alpha, X_groups, family


# True where every n_q-qubit block of the index holds a physical state (< d)
# What the program returned, and how far the circuit's own numbers sit from it.
# The split is an equality constraint, sum_G X_G = M, so the residual below is solver round-off and not an
# approximation; the lattice margin min_G min_Delta Delta^T X_G Delta is the quantity the exactness condition
# is about, and it is >= 0 exactly when no group needs rescaling.
#   [Input]  M, X, family, lattice : as returned by lp_decomposer
#            alpha                 : the program's optimum, exp(sum_G t_G / h_bar)
#            lambdas               : the rescalings the circuit actually encoded, one per group, read off the
#                                    diagonals after they were built rather than taken from the program
#            t_lp                  : wall time of the program, printed when it was measured
#   [Output] nothing is returned; a report is printed. Four numbers are worth reading. The residual says the
#            split reproduces M. The margin says whether every piece is a contraction where the register can
#            evaluate it. alpha is what the dilation costs, alpha^2 in shots. The disagreement between alpha
#            and prod lambda is an independent check of the whole chain, since the two are computed from
#            different objects, and anything above ~1e-9 means the encoding and the program have parted.
def lp_report(M, X, family, lattice, alpha, lambdas, t_lp=None):
    n_time = len(M)

    # sum_G X_G = M, imposed as an equality; the residual is round-off
    rebuilt = np.zeros_like(M)
    for G in family:
        for a, i in enumerate(G):
            for b in range(a, len(G)):
                rebuilt[i, G[b]] += X[G][a, b]
    residual = float(np.abs(np.triu(rebuilt - np.triu(M))).max())

    # Delta^T X_G Delta over the blip lattice: >= 0 is "this piece loses nothing"
    margins = []
    for G in family:
        v = lattice_vectors(lattice, len(G))
        margins.append(min(float(w @ X[G] @ w) for w in v) if v else 0.0)
    worst    = min(margins) if margins else 0.0
    n_ineq   = sum(len(lattice_vectors(lattice, len(G))) for G in family)
    prod_lam = float(np.prod(lambdas)) if len(lambdas) else 1.0
    gap      = abs(prod_lam - alpha) / max(alpha, 1e-300)

    print('=====================< Grouping Program >=====================')
    print(f'      Blip matrix : {n_time} x {n_time}')
    print(f'           Groups : {len(family)}  (sizes {sorted({len(G) for G in family})})')
    print(f'     Inequalities : {n_ineq}')
    if t_lp is not None:
        print(f'       Solve time : {t_lp:.3f} s')
    print(f'  Split residual  : {residual:.3e}   max |sum_G X_G - M|')
    print(f'   Lattice margin : {worst:+.3e}   min_G min_Delta Delta^T X_G Delta')
    print(f'    Rescaling (a) : {alpha:.12g}   program optimum')
    print(f'    Rescaling (b) : {prod_lam:.12g}   prod of the encoded lambdas')
    print(f'       Difference : {gap:.3e}   (a) against (b)')
    print(f'      Shot factor : {alpha ** 2:.6g}   alpha^2, what the dilation costs')
    print('==============================================================')


def physical_mask(n_block, n_q, d):
    x    = np.arange(2 ** (n_block * n_q))
    mask = np.ones(len(x), dtype=bool)
    for p in range(n_block):
        mask &= ((x >> (p * n_q)) & (2 ** n_q - 1)) < d
    return mask


# Uniformly controlled Ry whose ancilla-|0> block is diag(sigma), 0 <= sigma <= 1 (qargs: [ancilla] + controls)
def ucry_encoding(sigma):
    return UCRYGate([float(t) for t in 2 * np.arccos(np.clip(sigma, 0.0, 1.0))])

# ==============================================================================================================
# 3b. The Sz.-Nagy construction  (IF='sz_nagy')
#   baseline: one block dilation per coefficient, rescaled by its largest singular value
# ==============================================================================================================


# Hermitian matrix square root (negative eigenvalues clipped)
def sqrtm(M):
    evals, evecs = np.linalg.eigh(M)
    return evecs @ np.diag(np.sqrt(np.clip(evals, 0, None))) @ evecs.conj().T


# Sz.-Nagy dilation [[A, sqrt(1-AA^dag)], [sqrt(1-A^dag A), -A^dag]] of a (rescaled) contraction A
def dilation(IF, N_qb):
    contract_ratio = max(1.0, np.linalg.svd(IF, compute_uv=False).max())
    IF    = IF / contract_ratio
    IF_dg = IF.conj().T
    I     = gen_identity(N_qb).to_matrix()
    return np.block([[IF, sqrtm(I - IF @ IF_dg)], [sqrtm(I - IF_dg @ IF), -IF_dg]])


# Exact synthesis of U; random single-qubit twists cure qs_decomposition panics and silent inaccuracy
def synthesize_unitary(U, atol=1e-9, N_retry=8, seed=0, N_probe=2):
    N_qb   = int(np.log2(len(U)))
    rng    = np.random.default_rng(seed)
    probes = []
    for _ in range(N_probe):
        vec = rng.normal(size=2 ** N_qb) + 1j * rng.normal(size=2 ** N_qb); vec /= np.linalg.norm(vec)
        probes.append((vec, U @ vec))

    def error_of(circuit):                                                     # phase-free error on probe states
        worst = 0.0
        for vec, ref in probes:
            got   = Statevector(vec).evolve(circuit).data
            phase = np.vdot(ref, got)
            got   = got * np.exp(-1j * np.angle(phase)) if abs(phase) > 0 else got
            worst = max(worst, float(np.abs(got - ref).max()))
        return worst

    best, best_error, first_failure = None, np.inf, None
    for attempt in range(N_retry + 1):
        try:
            if attempt == 0:
                candidate = qs_decomposition(U)
            else:                                                              # U = (U R^dag) R
                twist = QuantumCircuit(N_qb)
                for q in range(N_qb):
                    twist.u(*rng.uniform(0, 2 * np.pi, 3), q)
                candidate = QuantumCircuit(N_qb)
                candidate.compose(twist, inplace=True)
                candidate.compose(qs_decomposition(U @ Operator(twist).data.conj().T), inplace=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:                                           # PanicException is a BaseException
            first_failure = first_failure or f'{type(exc).__name__}: {str(exc).splitlines()[0][:100]}'
            continue
        error = error_of(candidate)
        if error < best_error:
            best, best_error = candidate, error
        if error <= atol:
            return best
    if best is None:
        raise RuntimeError(f'qs_decomposition failed on {N_qb} qubits: {first_failure}')
    print(f'WARNING: {N_qb}-qubit synthesis accurate only to {best_error:.2e}')
    return best

# ==============================================================================================================
# 3c. The fused construction  (IF='fused', 'walsh_fourier')
#   reference floor: the whole IF as one diagonal, by Gray-code walk or Walsh series
# ==============================================================================================================


# Fast Walsh-Hadamard transform, h[j] = sum_k (-1)^popcount(j & k) a[k]
def fwht(a):
    a, h = np.array(a, dtype=complex), 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0, :] + a[:, 1, :], a[:, 0, :] - a[:, 1, :]], axis=1).reshape(-1)
        h *= 2
    return a


# diag(e^{i phi}) with exactly 2^q - 2 CNOTs (Gray-code walk over the Walsh terms)
def gray_diagonal(phi, cutoff=1e-14):
    q   = int(np.log2(len(phi)))
    a   = np.real(fwht(phi)) / len(phi)
    circ = QuantumCircuit(q, global_phase=a[0])
    for target in range(q):
        base = 1 << target
        if abs(a[base]) > cutoff:
            circ.rz(-2 * a[base], target)
        for i in range(1, base):
            circ.cx((i & -i).bit_length() - 1, target)                          # the bit gray(i) flips
            j = base | (i ^ (i >> 1))
            if abs(a[j]) > cutoff:
                circ.rz(-2 * a[j], target)
        if target:
            circ.cx(target - 1, target)                                        # restore the target
    return circ

# ==============================================================================================================
# 4. Readout and path sampling
#   state preparation, amplitudes, and the classical weights behind the selected basis
# ==============================================================================================================


# V|0> = uniform state over the first d of 2^n_q basis states (Hadamards when d = 2^n_q)
def uniform_prep(d, n_q):
    circ = QuantumCircuit(n_q)
    if d == 2 ** n_q:
        circ.h(range(n_q)); return circ
    amp = np.zeros(2 ** n_q); amp[:d] = 1 / np.sqrt(d)
    circ.append(StatePreparation(amp), range(n_q))
    return transpile(circ, basis_gates=COUNT_BASIS, optimization_level=1)


##########################################################################<<< Classic Modules

# Probability vector -> amplitude vector
def gen_amplitude(state):
    state = np.array(state, dtype=float)
    return np.sqrt(state / state.sum())


# Ordinal suffix (1st, 2nd, ...)
def rank_index(n):
    suffix = ['st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th', 'th']
    return 'th' if (n % 100) // 10 == 1 else suffix[n % 10]


# Bitstring -> X-gate operator
def state2op(bin_state):
    return UnitaryGate(SparsePauliOp(bin_state.replace('0', 'I').replace('1', 'X')))


# Classical path weight rho0 * prod <f'|U|f><b|U^dag|b'> * IF  (rho0_fb=None: segment weight, no initial state)
def path_weight(fwd, bwd, U, s_values, alphas, rho0_fb=None):
    weight = 1.0 + 0j if rho0_fb is None else complex(rho0_fb)
    for k in range(len(fwd) - 1):
        weight *= U[fwd[k + 1], fwd[k]] * np.conj(U[bwd[k + 1], bwd[k]])
    exponent = 0j
    for k_aft in range(len(fwd)):
        for k_bef in range(k_aft + 1):
            eta = alphas[k_aft, k_bef]
            exponent += (s_values[fwd[k_aft]] - s_values[bwd[k_aft]]) * \
                        (eta * s_values[fwd[k_bef]] - np.conj(eta) * s_values[bwd[k_bef]])
    return weight * np.exp(-exponent / h_bar)


# Metropolis walk over configurations (tuples of n_site values in [0, n_value)) with weight |w(config)|
#   [Input]  weight  : callable returning the (complex) path weight of one configuration; only |w| is sampled
#            n_site  : number of intermediate slices the configuration indexes
#            n_value : number of states each slice may take
#            n_sample, n_burn, seed : walk length, discarded head, and generator seed
#   [Output] the set of DISTINCT configurations the walk visited, not a weighted sample: it is used to choose
#            which basis states the readout keeps, so only membership matters and the visit counts are
#            discarded. Its size is at most n_sample and typically far smaller, and it grows with n_sample
#            until the walk stops finding new configurations
#            (on-the-fly path filtering, Sim & Makri, CPL 249, 224 (1996))
def metropolis_configs(weight, n_site, n_value, n_sample, seed=0, n_burn=200):
    rng    = np.random.default_rng(seed)
    config = tuple(int(x) for x in rng.integers(0, n_value, n_site))
    w_now  = abs(weight(config))
    visited = set()
    for step in range(n_burn + n_sample):
        trial = list(config); trial[rng.integers(n_site)] = int(rng.integers(n_value)); trial = tuple(trial)
        w_trial = abs(weight(trial))
        if w_now == 0 or rng.random() < w_trial / w_now:
            config, w_now = trial, w_trial
        if step >= n_burn and w_now > 0:
            visited.add(config)
    return visited

# ==============================================================================================================
# 5. class CXQUAPI
#   build the circuit (gen_IF_* -> gen_propagation_circuit) and read it (run)
# ==============================================================================================================

class CXQUAPI:
    # [IF]         'lp' (default), 'fused', 'sz_nagy', 'walsh_fourier'
    # [IF_setting] 'group_size' (lp, default 2), 'walsh_cutoff' (walsh_fourier)

    # --------------------------------------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------------------------------------
    def __init__(self,
                 H, S, dt, N_max_step,
                 N_memory   = 0,    N_state    = 2,    beta        = 1,
                 IF         = 'lp', IF_setting = None, verbose     = True):
        # Saving Values
        self.H              = H                                         # System Hamiltonian (SparsePauliOp)
        self.S              = S                                         # System-bath coupling operator, diagonal
        self.dt             = dt                                        # Time step
        self.N_max_step     = N_max_step                                # Number of time steps N
        self.N_memory       = N_memory                                  # Memory span (N for full memory)
        self.N_state        = N_state                                   # Number of physical system states d
        self.beta           = beta                                      # 1 / (k_B T)
        self.IF             = IF                                        # IF construction
        self.IF_setting     = IF_setting if IF_setting is not None else {}
        self.verbose        = verbose                                   # print the banner, the program report and the RDM
        assert IF in IF_Methods, f'IF must be one of {list(IF_Methods)}'

        # Auto-Loaded Values
        self.N_qb_state     = int(np.ceil(np.log2(N_state)))            # Qubits per slice per branch
        self.N_forward      = (N_max_step + 1) * self.N_qb_state        # Qubits of the forward chain
        self.N_ancilla      = 0                                         # IF ancillas, set by gen_IF_circuit
        self.lambdas        = None                                      # Rescaling of each IF group
        self.alpha_lp       = None                                      # LP optimum of prod lambda ('lp')
        self.groups         = None                                      # IF groups ('lp')
        self.cx_breakdown   = None                                      # CNOTs of the IF circuit parts
        self.p_postselect   = None                                      # P(ancillas = 0) per column
        self.p_readout      = None                                      # Readout probability per diagonal element
        self.RDM_raw        = None                                      # Unnormalized readout amplitudes
        self.M              = None                                      # Blip matrix of the IF ('lp')
        self.t_lp           = None                                      # Seconds the LP took ('lp')


    # Bath spectral density J(w); Ohmic by default, override per system
    def spectral_density(self, w):
        w_c, Kondo = 1, 1
        return 0.5 * np.pi * h_bar * Kondo * w * np.exp(-abs(w) / w_c)


    # Physical position values s_i (dummy register states dropped)
    def gen_s_values(self):
        return np.real(z_diagonal(SparsePauliOp.simplify(self.S)))[:self.N_state]


    # Bare propagation: copy each slice with CNOTs and apply exact U (forward) / U^dag (backward)
    def gen_system_circuit(self):
        circuit  = QuantumCircuit(2 * self.N_forward)
        H_matrix = self.H.to_matrix() if hasattr(self.H, 'to_matrix') else np.asarray(self.H)
        U_fwd    = UnitaryGate(expm(-1j * H_matrix * self.dt / h_bar), label='U')
        U_bwd    = UnitaryGate(expm(+1j * H_matrix * self.dt / h_bar), label='U_dag')
        n_q, N   = self.N_qb_state, self.N_max_step
        for step in range(N):
            for q in range(n_q):
                circuit.cx(n_q * step + q, n_q * (step + 1) + q)
                circuit.cx(n_q * (N - step) + q + self.N_forward, n_q * (N - step - 1) + q + self.N_forward)
            circuit.append(U_fwd, range(n_q * (step + 1), n_q * (step + 2)))
            circuit.append(U_bwd, range(n_q * (N - step - 1) + self.N_forward, n_q * (N - step) + self.N_forward))
        return circuit


    # IF exponent of every (k_aft, k_bef) term as a Z-only SparsePauliOp on its local qubits
    #   [Output] {(k_aft, k_bef): (IF_coef, qargs)} with k_aft >= k_bef, one entry per influence coefficient
    #            that survives the memory span. IF_coef is the exponent of that term, a SparsePauliOp built
    #            only from I and Z, so it is diagonal and the whole functional commutes term by term; its real
    #            part is the contraction the dilation must carry and its imaginary part the phase, which is
    #            synthesized separately and needs no ancilla. qargs lists the qubits the term acts on, four
    #            per term in general (forward and backward blocks of slices k_aft and k_bef) and two when
    #            k_aft == k_bef
    def gen_IF_coefs(self):
        positions = gen_expanded_position(self.N_qb_state, self.S)
        alphas    = alpha_generator(self.dt, self.N_memory, self.N_max_step, self.beta, self.spectral_density)
        n_q, IF_coefs = self.N_qb_state, {}
        slice_qubits = lambda k: list(range(k * n_q, (k + 1) * n_q)) + \
                                 list(range(self.N_forward + k * n_q, self.N_forward + (k + 1) * n_q))
        for k_aft in range(self.N_max_step + 1):
            for k_bef in range(0 if k_aft < self.N_memory else k_aft - self.N_memory, k_aft + 1):
                eta = alphas[k_aft, k_bef]
                if k_bef == k_aft:
                    coef  = -(positions[0][0] - positions[0][1]) @ (eta * positions[0][0] - np.conj(eta) * positions[0][1]) / h_bar
                    qargs = slice_qubits(k_aft)
                else:
                    coef  = -(positions[1][0] - positions[1][1]) @ (eta * positions[1][2] - np.conj(eta) * positions[1][3]) / h_bar
                    qargs = slice_qubits(k_aft) + slice_qubits(k_bef)
                IF_coefs[(k_aft, k_bef)] = (SparsePauliOp.simplify(coef), qargs)
        return IF_coefs

    # --------------------------------------------------------------------------------------------------------
    # Influence functional
    # --------------------------------------------------------------------------------------------------------


    # IF circuit of the selected construction
    def gen_IF_circuit(self):
        builder = {'lp': self.gen_IF_lp, 'fused': self.gen_IF_fused,
                   'sz_nagy': self.gen_IF_sz_nagy, 'walsh_fourier': self.gen_IF_walsh}[self.IF]
        return builder()


    # Phase of every IF term fused into one ancilla-free diagonal unitary
    def gen_IF_phase(self, IF_coefs, n_qubit_total):
        N_path   = 2 * self.N_forward
        im_total = SparsePauliOp.from_list([('I' * N_path, 0.0)])
        for coef, qargs in IF_coefs.values():
            im_total = im_total + embed_local(SparsePauliOp(coef.paulis, coef.coeffs.imag), qargs, list(range(N_path)))
        im_total = im_total.simplify()
        circ = QuantumCircuit(n_qubit_total)
        circ.append(PauliEvolutionGate(SparsePauliOp(im_total.paulis, -im_total.coeffs.real), time=1.0), range(N_path))
        return circ


    # 'sz_nagy': each term dilated separately and synthesized generically (reference construction)
    def gen_IF_sz_nagy(self):
        IF_coefs = self.gen_IF_coefs()
        N_path, gates, lambdas = 2 * self.N_forward, [], []
        for coef, qargs in IF_coefs.values():
            IF_matrix = expm(coef.to_matrix())
            lambdas.append(max(1.0, np.linalg.svd(IF_matrix, compute_uv=False).max()))
            gates.append((synthesize_unitary(dilation(IF_matrix, coef.num_qubits)), qargs + [N_path + len(gates)]))
        circuit = QuantumCircuit(N_path + len(gates))
        for gate, gate_qargs in gates:
            circuit.compose(gate, qubits=gate_qargs, inplace=True)
        self.N_ancilla, self.lambdas = len(gates), lambdas
        return circuit


    # 'walsh_fourier': same fused diagonal, one parity ladder per Walsh string (reference construction)
    def gen_IF_walsh(self):
        IF_coefs = self.gen_IF_coefs()
        N_path   = 2 * self.N_forward
        phi, lam = self.gen_IF_diagonal_phases(IF_coefs)
        q        = N_path + 1
        a        = np.real(fwht(phi)) / len(phi)
        keep     = np.flatnonzero(np.abs(a) > self.IF_setting.get('walsh_cutoff', 1e-12))
        if keep.size == 0:
            keep = np.array([0])
        z_bits   = ((keep[:, None] >> np.arange(q)[None, :]) & 1).astype(bool)
        op       = SparsePauliOp(PauliList.from_symplectic(z_bits, np.zeros_like(z_bits)), -a[keep])
        circuit  = QuantumCircuit(q)
        circuit.h(N_path)
        circuit.append(PauliEvolutionGate(op, time=1.0), range(q))
        circuit.h(N_path)
        self.N_ancilla, self.lambdas = 1, [lam]
        return transpile(circuit, optimization_level=0)


    # Fused IF as one diagonal: I_+- = e^{i arg sigma} e^{+-i arccos|sigma|}, Gray-code synthesis, 1 ancilla
    def gen_IF_diagonal_phases(self, IF_coefs):
        N_path = 2 * self.N_forward
        total  = SparsePauliOp.from_list([('I' * N_path, 0.0)])
        for coef, qargs in IF_coefs.values():
            total = total + embed_local(coef, qargs, list(range(N_path)))
        sigma = np.exp(z_diagonal(SparsePauliOp.simplify(total)))
        lam   = max(1.0, float(np.abs(sigma).max()))
        sigma = sigma / lam
        phase, rot = np.angle(sigma), np.arccos(np.clip(np.abs(sigma), 0.0, 1.0))
        return np.concatenate([phase + rot, phase - rot]), lam                 # ancilla = highest qubit

    def gen_IF_fused(self):
        IF_coefs   = self.gen_IF_coefs()
        N_path     = 2 * self.N_forward
        phi, lam   = self.gen_IF_diagonal_phases(IF_coefs)
        circuit    = QuantumCircuit(N_path + 1)
        circuit.h(N_path)
        circuit.compose(gray_diagonal(phi), range(N_path + 1), inplace=True)
        circuit.h(N_path)
        self.N_ancilla, self.lambdas = 1, [lam]
        return transpile(circuit, basis_gates=COUNT_BASIS, optimization_level=0)


    # 'lp': modulus split over LP-optimal groups (one UCRY + ancilla each), phases fused (Letter Eq. 4)
    def gen_IF_lp(self):
        IF_coefs = self.gen_IF_coefs()
        N_path   = 2 * self.N_forward
        g        = self.IF_setting.get('group_size', 2)
        re_part  = {kk: SparsePauliOp(c.paulis, c.coeffs.real) for kk, (c, _) in IF_coefs.items()}
        qargs    = {kk: q for kk, (_, q) in IF_coefs.items()}

        alphas = alpha_generator(self.dt, self.N_memory, self.N_max_step, self.beta, self.spectral_density)
        M      = blip_matrix(alphas, self.N_memory)
        t0 = time.time()
        self.alpha_lp, X, family = lp_decomposer(M, blip_lattice(self.gen_s_values()), g)
        self.t_lp = time.time() - t0                                    # wall time of the grouping program
        self.M    = M                                                   # blip matrix the program was given

        herm = QuantumCircuit(N_path + len(family))
        lambdas = []
        for anc, G in enumerate(family):
            group_qubits = [q for t in G for q in qargs[(t, t)]]
            generator    = SparsePauliOp.from_list([('I' * len(group_qubits), 0.0)])
            for a, i in enumerate(G):
                for b in range(a, len(G)):
                    j = G[b]
                    if X[G][a, b] != 0.0:
                        key = (max(i, j), min(i, j))
                        generator = generator + (X[G][a, b] / M[i, j]) * embed_local(re_part[key], qargs[key], group_qubits)
            sigma = np.exp(z_diagonal(SparsePauliOp.simplify(generator)).real)
            phys  = physical_mask(2 * len(G), self.N_qb_state, self.N_state)
            lam   = max(1.0, float(sigma[phys].max()))
            lambdas.append(lam)
            herm.append(ucry_encoding(np.minimum(sigma / lam, 1.0)), [N_path + anc] + group_qubits)

        phase = transpile(self.gen_IF_phase(IF_coefs, herm.num_qubits), basis_gates=COUNT_BASIS, optimization_level=0)
        herm  = transpile(herm, basis_gates=COUNT_BASIS, optimization_level=0)
        self.N_ancilla, self.lambdas, self.groups = len(family), lambdas, family
        self.cx_breakdown = {'hermitian': herm.count_ops().get('cx', 0), 'unitary': phase.count_ops().get('cx', 0)}
        if self.verbose:
            lp_report(M, X, family, blip_lattice(self.gen_s_values()), self.alpha_lp, lambdas, self.t_lp)
        return phase.compose(herm)

    # --------------------------------------------------------------------------------------------------------
    # Full circuit
    # --------------------------------------------------------------------------------------------------------


    # Bare propagation followed by the IF circuit
    def gen_propagation_circuit(self):
        system_circ = self.gen_system_circuit()
        IF_circ     = self.gen_IF_circuit()
        circuit     = QuantumCircuit(2 * self.N_forward + self.N_ancilla)
        circuit.compose(system_circ, qubits=range(system_circ.num_qubits), inplace=True)
        circuit.compose(IF_circ, qubits=range(circuit.num_qubits), inplace=True)
        return circuit

    # --------------------------------------------------------------------------------------------------------
    # Readout
    # --------------------------------------------------------------------------------------------------------


    # Qubits of the intermediate slices 1..N-1 (fwd block, bwd block per slice): these are summed
    def gen_readout_geometry(self):
        n_q, summed = self.N_qb_state, []
        for k in range(1, self.N_max_step):
            summed += list(range(k * n_q, (k + 1) * n_q))
            summed += list(range(self.N_forward + k * n_q, self.N_forward + (k + 1) * n_q))
        return summed


    # Basis index of the readout string for RDM[sN_f, sN_b]; None if two assignments to one block disagree
    def gen_readout_index(self, sN_f, sN_b, s0_f, s0_b):
        n_q, assign = self.N_qb_state, {}
        for start, value in ((0, int(s0_f, 2)), (self.N_forward - n_q, sN_f),
                             (self.N_forward, int(s0_b, 2)), (2 * self.N_forward - n_q, sN_b)):
            if assign.setdefault(start, value) != value:
                return None
        return sum(value << start for start, value in assign.items())


    # Input label: |sN_b> on the backward final block, |s0_f> on the forward first block, ancillas |0>
    def gen_init_label(self, sN_b, s0_f):
        label = ['0'] * (2 * self.N_forward + self.N_ancilla)
        label[self.N_ancilla:self.N_ancilla + self.N_qb_state] = list(bin(sN_b)[2:].zfill(self.N_qb_state))
        label[len(label) - self.N_qb_state:] = list(s0_f.zfill(self.N_qb_state))
        return ''.join(label)


    # Path sum over the intermediate slices: V^dag per register, V|0> uniform over the d physical states
    def gen_path_sum(self, n_qubit, summed):
        n_q, circ = self.N_qb_state, QuantumCircuit(n_qubit)
        V_dag = uniform_prep(self.N_state, n_q).inverse()
        for start in range(0, len(summed), n_q):
            circ.compose(V_dag, summed[start:start + n_q], inplace=True)
        return circ


    # Readout amplitude restricted to a set of intermediate configurations: <u_S|psi>, |u_S> uniform over S
    #   config = tuple of local state indices, one per summed register block (gen_readout_geometry order)
    def gen_restricted_amplitude(self, psi_pre, base_index, configs):
        summed, n_q = self.gen_readout_geometry(), self.N_qb_state
        starts = summed[::n_q]
        amp = sum(psi_pre[base_index + sum(v << s for v, s in zip(cfg, starts))] for cfg in configs)
        return amp / np.sqrt(len(configs))

    # --------------------------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------------------------
    AER_MIN_QUBITS = 16                                                        # Aer beats Statevector from here


    # Configuration banner, printed once per run so a log says what produced the numbers under it
    #   [Output] nothing is returned; the settings the circuit was built from are printed. Qubits are split
    #            into the two path chains and the IF ancillas, one per group, and the CNOT line is the
    #            contractive layer against the phase layer, which is the split the gate counts are quoted in.
    def print_banner(self, n_qubit):
        cx = self.cx_breakdown or {}
        print(f'=======================< CX-QuAPI >=======================\n'
              f' Number of States : {self.N_state}\n'
              f'    Max Time (dt) : {self.N_max_step * self.dt:.6g} ({self.dt})\n'
              f'      Memory Span : {self.N_memory}\n'
              f'         Max Step : {self.N_max_step}\n'
              f'             Beta : {self.beta:.3f}\n'
              f'               IF : {self.IF}'
              + (f" (group size {self.IF_setting.get('group_size', 2)})" if self.IF == 'lp' else '') + '\n'
              f'           Qubits : {n_qubit} ({2 * self.N_forward} path + {self.N_ancilla} ancilla)\n'
              f'            CNOTs : {sum(cx.values())} '
              f"({cx.get('hermitian', 0)} contraction + {cx.get('unitary', 0)} phase)\n"
              f'        Rescaling : {self.alpha_lp if self.alpha_lp is not None else 1.0:.12g}\n'
              f'==========================================================')


    # RDM[sN_f, sN_b] read from the statevector; readout_set={(a, b): configs} restricts the path sum
    #   [Input] backend     : 'auto' | 'aer' | 'statevector'
    #           max_threads : Aer threads
    #           s0_f, s0_b  : the initial forward and backward indices; they must agree for the trace used
    #                         to normalize to be the physical one
    #           readout_set : {(a, b): configs} to sum only the listed intermediate configurations instead of
    #                         all d^(2(N-1)) of them. This is the truncated readout: the result is then an
    #                         approximation of the path sum, biased rather than noisy
    #   [Output] RDM : the reduced density matrix at t = N dt, trace-normalized. Normalizing divides out the
    #                  circuit's global phase and the post-selection weight at once, so the returned matrix is
    #                  directly comparable with a classical path sum; the diagonal is forced real afterwards.
    #                  The unnormalized amplitudes are kept in self.RDM_raw, and it is those, not the returned
    #                  matrix, that carry the dilution d^(-2(N-1)) the shot count is set by.
    #            Also set on the object as a side effect:
    #                  self.RDM_raw      the unnormalized readout amplitudes
    #                  self.p_postselect P(all IF ancillas = 0) per column, the fraction of runs kept
    #                  self.p_readout    single-shot probability of each diagonal element, which with the
    #                                    rescaling fixes the shots the element costs
    def run(self, s0_f='0', s0_b='0', backend='auto', max_threads=None, readout_set=None):
        time_start   = time.time()
        circuit_prop = self.gen_propagation_circuit()
        N_path       = 2 * self.N_forward
        summed       = self.gen_readout_geometry()
        simulator    = self.gen_simulator(backend, N_path + self.N_ancilla, max_threads)
        RDM          = np.zeros((self.N_state, self.N_state), dtype=complex)
        p_postselect, p_readout = [], []

        for sN_b in range(self.N_state):
            label = self.gen_init_label(sN_b, s0_f)
            if simulator is not None:
                psi_pre, psi_post = self.run_aer(simulator, circuit_prop, summed, label)
            else:
                psi_pre  = Statevector.from_label(label).evolve(circuit_prop).data
                psi_post = Statevector(psi_pre).evolve(self.gen_path_sum(N_path + self.N_ancilla, summed)).data \
                           if summed else psi_pre
            p_postselect.append(float(np.sum(np.abs(psi_pre[:2 ** N_path]) ** 2)))
            for sN_f in range(self.N_state):
                idx = self.gen_readout_index(sN_f, sN_b, s0_f, s0_b)
                if idx is None:
                    value = 0.0
                elif readout_set is not None and summed:
                    value = self.gen_restricted_amplitude(psi_pre, idx, readout_set[(sN_f, sN_b)])
                    scale = np.sqrt(len(readout_set[(sN_f, sN_b)]))            # undo 1/sqrt(R): sets differ per element
                else:
                    value, scale = psi_post[idx], 1.0
                RDM[sN_f, sN_b] = value * scale                                # complex: global phase is common
                if sN_f == sN_b:
                    p_readout.append(float(abs(value) ** 2))                   # one shot reads the raw amplitude

        trace = np.trace(RDM)
        if abs(trace) < 1e-12:
            raise ValueError(f'RDM trace {trace:.3e} too small to normalize (use s0_f == s0_b)')
        self.RDM_raw, self.p_postselect, self.p_readout = RDM.copy(), p_postselect, p_readout
        RDM /= trace                                                           # removes the circuit's global phase
        RDM[np.diag_indices(self.N_state)] = np.real(np.diag(RDM))
        if self.verbose:
            self.print_banner(N_path + self.N_ancilla)
            print('Time\t ' + '\t '.join(f'Re({i}, {j})' for i in range(self.N_state)
                                          for j in range(self.N_state)) + '\t Tr(rho)')
            print(f'{self.N_max_step * self.dt:.3f}\t '
                  + '\t '.join(f'{RDM[i, j].real:.7f}' for i in range(self.N_state)
                                for j in range(self.N_state))
                  + f'\t {RDM.trace().real:.7f}')
            print(f"P(post-select) {min(p_postselect):.3e}-{max(p_postselect):.3e}   "
                  f"P(readout) {min(p_readout):.3e}-{max(p_readout):.3e}   "
                  f"{time.time() - time_start:.2f}s")
        return RDM

    # Everything this object knows about the run, in one file.
    # The circuit is the only thing it cannot hold -- that is rebuilt on demand to be counted -- so a
    # report written after run() is a complete record of the calculation: what was asked for, what the
    # grouping program returned, what the circuit cost, and what came back from the measurement.
    #   [Input]  path    : '.json' writes a machine-readable record, anything else a readable report
    #            circuit : also transpile and count the propagation circuit (slow for large N)
    #            levels  : optimization levels to count it at; (0,) is the structural count
    #   [Output] the record, as a dictionary, whether or not a path was given. Four sections. 'config' is what
    #            was asked for. 'grouping' is what the program returned -- the groups, the rescaling, the per
    #            group lambdas, the ancilla count, the CNOT split and the blip matrix it was given -- which is
    #            everything needed to rebuild the split without re-solving. 'circuit' is counted rather than
    #            stored, since the circuit itself is the one object too large to keep, and it carries a CNOT
    #            count and depth per optimization level. 'result' holds both the raw amplitudes and their
    #            trace-normalized form, because the raw ones carry the dilution d^(-2(N-1)) that sets the shot
    #            count and the normalized ones are what compares against a classical path sum
    def dump(self, path=None, circuit=True, levels=(0,)):
        def plain(x):                                                          # numpy -> json
            if isinstance(x, np.ndarray):   return [plain(v) for v in x.tolist()]
            if isinstance(x, (list, tuple)):return [plain(v) for v in x]
            if isinstance(x, dict):         return {str(k): plain(v) for k, v in x.items()}
            if isinstance(x, complex):      return {'re': float(x.real), 'im': float(x.imag)}
            if isinstance(x, (np.integer,)):return int(x)
            if isinstance(x, (np.floating,)):return float(x)
            return x

        circ = self.gen_propagation_circuit() if circuit else None     # sets groups, alpha_lp, lambdas

        rec = {'config': {'N_state': self.N_state, 'N_max_step': self.N_max_step,
                          'N_memory': self.N_memory, 'dt': self.dt, 'beta': self.beta,
                          'IF': self.IF, 'IF_setting': plain(self.IF_setting),
                          'N_qb_state': self.N_qb_state, 'N_forward': self.N_forward},
               'grouping': {'groups': plain(self.groups), 'alpha_lp': plain(self.alpha_lp),
                            'lambdas': plain(self.lambdas), 'N_ancilla': self.N_ancilla,
                            'cx_breakdown': plain(self.cx_breakdown), 't_lp': plain(self.t_lp),
                            'M': plain(self.M)},
               'readout': {'p_postselect': plain(self.p_postselect),
                           'p_readout': plain(self.p_readout)},
               'result': {}}

        if circ is not None:                                               # counted, not stored
            rec['circuit'] = {'qubits': int(circ.num_qubits)}
            for lv in levels:
                tc = transpile(circ, basis_gates=COUNT_BASIS, optimization_level=lv)
                rec['circuit'][f'cx_O{lv}'] = int(tc.count_ops().get('cx', 0))
                rec['circuit'][f'depth_O{lv}'] = int(tc.depth())
            rec['circuit']['cx'] = rec['circuit'][f'cx_O{levels[0]}']
            rec['circuit']['depth'] = rec['circuit'][f'depth_O{levels[0]}']

        if self.RDM_raw is not None:
            raw = np.asarray(self.RDM_raw)
            rec['result']['RDM_raw'] = plain(raw)
            tr = np.trace(raw)
            if abs(tr) > 1e-12:
                rec['result']['RDM'] = plain(raw / tr)

        if path is not None:
            if str(path).endswith('.json'):
                with open(path, 'w') as fh:
                    json.dump(rec, fh, indent=1)
            else:
                with open(path, 'w') as fh:
                    for section, body in rec.items():
                        fh.write(f'[{section}]\n')
                        for k, v in body.items():
                            fh.write(f'  {k:<14} {v}\n')
                        fh.write('\n')
            print(f'{path} written')
        return rec

    # AerSimulator(statevector) when worth its start-up cost, else None
    def gen_simulator(self, backend, N_qubit, max_threads):
        if backend == 'statevector' or (backend == 'auto' and N_qubit < self.AER_MIN_QUBITS):
            return None
        try:
            from qiskit_aer import AerSimulator
        except ImportError:
            if backend == 'aer':
                raise
            return None
        options = {'method': 'statevector'}
        if max_threads is not None:
            options['max_parallel_threads'] = max_threads
        return AerSimulator(**options)


    # One Aer run per column: statevector before and after the path sum
    def run_aer(self, simulator, circuit_prop, summed, label):
        N_qubit = circuit_prop.num_qubits
        circuit = QuantumCircuit(N_qubit)
        for c, bit in enumerate(label):
            if bit == '1':
                circuit.x(N_qubit - 1 - c)
        circuit.compose(circuit_prop, qubits=range(N_qubit), inplace=True)
        circuit.save_statevector(label='pre')
        if summed:
            circuit.compose(self.gen_path_sum(N_qubit, summed), inplace=True)
        circuit.save_statevector(label='post')
        data = simulator.run(transpile(circuit, simulator, optimization_level=0)).result().data(0)
        return np.asarray(data['pre']), np.asarray(data['post'])


    # Density-matrix run for a Qiskit Aer NoiseModel; RDM[a,b] = rho[idx_a, idx_b] / sqrt(rho[idx_b, idx_b])
    def run_noisy(self, noise_model=None, basis_gates=None, coupling_map=None, s0_f='0', s0_b='0'):
        from qiskit_aer import AerSimulator
        time_start   = time.time()
        circuit_prop = self.gen_propagation_circuit()
        N_path       = 2 * self.N_forward
        N_qubit      = N_path + self.N_ancilla
        summed       = self.gen_readout_geometry()
        if basis_gates is None and noise_model is not None:
            basis_gates = noise_model.basis_gates
        options = {'method': 'density_matrix'}
        for key, value in (('noise_model', noise_model), ('basis_gates', basis_gates), ('coupling_map', coupling_map)):
            if value is not None:
                options[key] = value
        simulator = AerSimulator(**options)

        RDM, p_postselect = np.zeros((self.N_state, self.N_state), dtype=complex), []
        for sN_b in range(self.N_state):
            circuit = QuantumCircuit(N_qubit)
            for c, bit in enumerate(self.gen_init_label(sN_b, s0_f)):
                if bit == '1':
                    circuit.x(N_qubit - 1 - c)
            circuit.compose(circuit_prop, qubits=range(N_qubit), inplace=True)
            if summed:
                circuit.compose(self.gen_path_sum(N_qubit, summed), inplace=True)
            circuit.save_density_matrix(label='rho')
            rho    = np.asarray(simulator.run(transpile(circuit, simulator, optimization_level=0)).result().data(0)['rho'])
            rho_ps = rho[:2 ** N_path, :2 ** N_path]                           # ancillas post-selected in |0>
            p_postselect.append(float(np.real(np.trace(rho_ps))))
            idx_b  = self.gen_readout_index(sN_b, sN_b, s0_f, s0_b)
            if idx_b is None:
                continue
            norm = np.sqrt(np.real(rho_ps[idx_b, idx_b]))
            if norm < 1e-14:
                raise ValueError(f'column {sN_b}: reference population {rho_ps[idx_b, idx_b]:.3e} lost to noise')
            for sN_f in range(self.N_state):
                idx_a = self.gen_readout_index(sN_f, sN_b, s0_f, s0_b)
                RDM[sN_f, sN_b] = 0.0 if idx_a is None else rho_ps[idx_a, idx_b] / norm
        RDM /= np.trace(RDM)
        self.p_postselect = p_postselect
        print(f"Noisy RDM (IF='{self.IF}', P(post-select) {min(p_postselect):.3e}-{max(p_postselect):.3e}, "
              f"{time.time() - time_start:.2f}s)")
        return RDM

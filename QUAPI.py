from scipy.integrate import quad
from scipy.linalg import expm
import numpy as np
import os

##########################################################################<<< Parameters

k_B     = 1
h_bar   = 1

##########################################################################<<< Modules

# Influence Functional Coefficient Calculator
def alpha_generator(dt, N_memory, N_max_step, beta, spectral_density):
    alpha_matrix    = np.zeros((N_max_step+1, N_max_step+1), dtype=complex)
    t               = dt * N_max_step

    # Functions for Integrand
    def integrand(w, k_bef, k_aft): # when omega is far from zero
        J_w         = spectral_density(w)
        #common_term = 2 / (1 - np.exp(-1 * h_bar * w * beta))
        common_term = 2 / (1 - np.exp(np.clip(-1 * h_bar * w * beta, -700, 700)))
        extra_term  = 1

        # kk'
        if 0 < k_bef and k_bef < k_aft and k_aft < N_max_step:
            extra_term *= (2 / np.pi) * (np.sin(w * dt / 2) ** 2) * np.exp(-1.0j * w * dt * (k_aft - k_bef))

        # kk
        if k_aft == k_bef and 0 < k_bef and k_bef < N_max_step:
            extra_term *= (0.5 / np.pi) * (1 - np.exp(-1.0j * w * dt))

        # N0
        if k_bef == 0 and k_aft == N_max_step:
            extra_term *= (2 / np.pi) * (np.sin(w * dt / 4) ** 2) * np.exp(-1.0j * w * (t - dt / 2))

        # 00 or NN
        if k_bef == k_aft and (k_aft == 0 or k_aft == N_max_step):
            extra_term *= (0.5 / np.pi) * (1 - np.exp(-1.0j * w * dt / 2))

        # k0
        if k_bef == 0 and 0 < k_aft and k_aft < N_max_step:
            extra_term *= (2 / np.pi) * (np.sin(w * dt / 4) * np.sin(w * dt / 2)) * np.exp(-1.0j * w * (k_aft * dt - dt / 4))

        # Nk
        if k_aft == N_max_step and 0 < k_bef and k_bef < N_max_step:
            extra_term *= (2 / np.pi) * (np.sin(w * dt / 4) * np.sin(w * dt / 2)) * np.exp(-1.0j * w * (t - k_bef * dt - dt / 4))

        return J_w * common_term * extra_term / (w ** 2)

    # Integration
    def integrator(integrand, k_bef, k_aft):
        val_re, _ = quad(lambda w: np.real(integrand(w, k_bef, k_aft)), -np.inf, np.inf)
        val_im, _ = quad(lambda w: np.imag(integrand(w, k_bef, k_aft)), -np.inf, np.inf)
        value = val_re + 1.0j * val_im

        return value

    for k_aft in range(N_max_step + 1):
        sum_min = 0 if k_aft < N_memory else k_aft - N_memory
        for k_bef in range(sum_min, k_aft + 1):
            alpha_matrix[k_aft, k_bef] = integrator(integrand, k_bef, k_aft)

    return alpha_matrix


# Alpha Value Printer
def alpha_printer(alphas):
    print('==========< IF Coefficients >==========')
    N_step, _   = alphas.shape
    for i in range(N_step):
        for j in range(i+1):
            print(f" {i}_{j} : {alphas[i,j]:5f}")


# Returns Rank Indices
def rank_index(n):
    ind_lst = ['st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th', 'th']
    if (n % 100) // 10 == 1:
        return 'th'
    else:
        return ind_lst[(n % 100) % 10]


# Change RDM into Paths & Weights
def RDM2Paths(RDM):
    N_state     = RDM.shape[0]
    Paths       = []
    Weights     = []
    for f in range(N_state):
        for b in range(N_state):
            Paths.append(np.array([f, b]))
            Weights.append(RDM[f, b])
    return np.array(Paths), np.array(Weights)


# Bare System Propagation
def bare_propagation(path, weight, Propagator, N_state):
    N_step      = int(len(path) // 2)
    new_paths   = []
    new_weights = []
    for N_forward in range(N_state):
        for N_backward in range(N_state):
            new_forward     = np.append(path[:N_step], [N_forward])
            new_backward    = np.append(path[N_step:], [N_backward])
            new_paths.append(np.append(new_forward, new_backward))
            prop_weight     = Propagator[path[N_step - 1], path[-1], N_forward, N_backward]
            new_weights.append(weight * prop_weight)
    return new_paths, new_weights


# Apply Influence Functioanl
def apply_IF(path, weight, positions, current_alphas, N_memory, N_step):
    len_path    = int(len(path) // 2)
    memory_step = min(N_memory, N_step)
    IF_coef     = 0
    for k_bef in range(memory_step + 1):
        alpha       = current_alphas[N_step - k_bef]
        IF_coef    += (positions[path[len_path - 1]] - positions[path[-1]]) * (alpha * positions[path[len_path - k_bef - 1]] - alpha.conjugate() * positions[path[-1 - k_bef]])
    IF_coef    /= h_bar
    IF_var      = np.exp(-IF_coef)
    return weight * IF_var


# Trace Paths
def trace_path(paths, weights, N_state):
    N_step  = int(paths.shape[1] // 2) - 1
    RDM     = np.zeros((N_state, N_state), dtype=complex)
    for N_path in range(len(weights)):
        path                            = paths[N_path]
        RDM[path[N_step], path[-1]]    += weights[N_path]
    return RDM


##########################################################################<<< Running Class

class FPF:
    def __init__(self,
                 RDM, H, S, t_max, dt,
                 N_memory=0, beta=1, cutoff=0, print_step=1, chkfile=None):
        # Input Values
        self.RDM                    = RDM                               # Initial RDM
        self.H                      = H                                 # Hamiltonian
        self.S                      = S                                 # Position Values
        self.t_max                  = t_max                             # Max Time
        self.dt                     = dt                                # Time Step
        self.N_memory                = N_memory                          # Memory Span
        self.beta                   = beta                              # 1 / (k_B * T)
        self.chkfile                = chkfile                           # Checkfile Save Directory
        self.cutoff                 = cutoff                            # Cutoff of OFPF
        self.print_step             = print_step                        # Step for Output Print

        # Load Values
        self.N_state                = len(S)                            # Number of States
        self.N_max_step             = int(t_max // dt)                  # Maximum Step Number
        self.paths, self.weights    = RDM2Paths(RDM)                    # Paths and Weights
        self.is_OFPF                = True if cutoff != 0 else False    # Whether Do On-the-Fly Path Filtering


    # Spectral Density (Ohmic by default). Every parameter it needs (cutoff frequency,
    # coupling strength, functional form, ...) lives inside this method. Override it
    # (e.g. via subclassing or instance-level reassignment) to swap in any other bath
    # spectral density J(w) without touching the rest of the class.
    def spectral_density(self, w):
        w_c     = 1     # Cutoff Angular Frequency
        Kondo   = 1     # Kondo Parameter

        return 0.5 * np.pi * h_bar * Kondo * w * np.exp(-abs(w) / w_c)


    # Generate Propagator Matrix (Only Forward, Backward as its adjoint)
    def gen_propagator(self):
        # P_ij: <j|exp(-iHdt/hbar)|i> ... Forward Propagator
        Forward     = expm(-1.0j * self.H * self.dt / h_bar)
        Propagator  = np.einsum('ik,jl->ijkl', Forward, Forward.conjugate().transpose())
        return Propagator


    # Propagate...
    def run(self):
        # Announcement
        print(f'================< FPF >================ \n'
              f' Number of States : {self.N_state}          \n'
              f'    Max Time (dt) : {self.t_max} ({self.dt})\n'
              f'      Memory Span : {self.N_memory}         \n'
              f'         Max Step : {self.N_max_step}       \n'
              f'             Beta : {self.beta:.3f}         \n'
              f'             OFPF : {self.is_OFPF} ({self.cutoff:.3e})\n'
              f'======================================= \n')

        # Initialize
        if self.chkfile is not None:
            os.makedirs(self.chkfile, exist_ok=True)
        alphas      = alpha_generator(self.dt, self.N_memory, self.N_max_step, self.beta, self.spectral_density)
        propagator  = self.gen_propagator()
        print("Time\t Re(0, 0)\t Re(1, 1)\t Re(0, 1)\t Tr(rho)")

        # Propagate
        for N_step in range(self.N_max_step + 1):
            local_alpha = alpha_generator(self.dt, self.N_memory, N_step, self.beta, self.spectral_density) # For End-point Dissipation, Output Weights
            new_paths   = []
            new_weights = [] # Mid-Point Dissipation
            end_weights = [] # End-Point Dissipation for Current Step

            # System Propagation
            if N_step:
                for N_path in range(len(self.weights)):
                    new_path, new_weight    = bare_propagation(self.paths[N_path], self.weights[N_path], propagator, self.N_state)
                    new_paths              += new_path
                    new_weights            += new_weight
                new_paths   = np.array(new_paths)
                new_weights = np.array(new_weights, dtype=np.complex128)
                end_weights = np.array(new_weights, dtype=np.complex128)
            else:
                new_paths   = np.array(self.paths)
                new_weights = np.array(self.weights, dtype=np.complex128)
                end_weights = np.array(self.weights, dtype=np.complex128)

            # Dissipation
            for N_path in range(len(new_weights)):
                end_weights[N_path] = apply_IF(new_paths[N_path], end_weights[N_path], self.S, local_alpha[N_step], self.N_memory, N_step)
                new_weights[N_path] = apply_IF(new_paths[N_path], new_weights[N_path], self.S, alphas[N_step], self.N_memory, N_step)

            # OFPF
            if self.is_OFPF:
                filtered_paths          = []
                filtered_new_weights    = []
                filtered_end_weights    = []
                for N_path in range(len(new_weights)):
                    if np.linalg.norm(new_weights[N_path]) > self.cutoff:
                        filtered_paths.append(new_paths[N_path])
                        filtered_new_weights.append(new_weights[N_path])
                        filtered_end_weights.append(end_weights[N_path])
                filtered_paths          = np.array(filtered_paths)
                filtered_new_weights    = np.array(filtered_new_weights, dtype=np.complex128)
                filtered_end_weights    = np.array(filtered_end_weights, dtype=np.complex128)
                new_paths               = filtered_paths
                new_weights             = filtered_new_weights
                end_weights             = filtered_end_weights

            # Merge
            if new_paths.shape[1] > 2 * self.N_memory and N_step != 0:
                len_path    = int(new_paths.shape[1] // 2)
                new_paths   = np.append(new_paths[:, 1:len_path], new_paths[:, len_path+1:], axis=1)
            unique_paths, inverse_paths = np.unique(new_paths, axis=0, return_inverse=True)
            unique_mid_weights          = np.zeros(len(unique_paths), dtype=np.complex128)
            unique_end_weights          = np.zeros(len(unique_paths), dtype=np.complex128)
            np.add.at(unique_mid_weights, inverse_paths, new_weights)
            np.add.at(unique_end_weights, inverse_paths, end_weights)

            # Print
            RDM = trace_path(unique_paths, unique_end_weights, self.N_state) # end_weights only for output
            if N_step % self.print_step == 0:
                print(f"{self.dt * N_step:.3f}\t {RDM[0, 0].real:.7f}\t {RDM[1, 1].real:.7f}\t {RDM[0, 1].real:.7f}\t {RDM.trace().real:.7f}")

            # Save
            self.paths      = unique_paths
            self.weights    = unique_mid_weights
            if self.chkfile is not None:
                np.save(f"{self.chkfile}/{np.around(N_step*self.dt, 3)}", RDM)

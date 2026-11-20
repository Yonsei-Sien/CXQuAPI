# CX-QuAPI audit: TLS-strong, N = 3, groups of 2 slices

Written by `cx_audit.py` on 2026-10-06 11:38 (0.7 s).

## Files

- `01_influence_coefficients.txt` — eta[k_aft, k_bef] (lower triangle), real part then imaginary part
- `02_blip_matrix.txt` — M with M_kk = Re eta_kk and M_kk' = Re eta_kk'/2; |IF| = exp(-Delta^T M Delta)
- `03_blip_matrix.json` — the same matrix as numbers
- `04_lattice_and_groups.txt` — physical positions, the blip lattice Lambda = {s_i - s_j}, and the groups of time slices
- `05_lp_solution.txt` — each group's piece X_G, its slack t_G and its rescaling lambda_G
- `06_lp_solution.json` — the same pieces as numbers
- `07_circuit.txt` — register layout and transpiled gate counts
- `08_circuit_diagram.txt` — the propagation circuit as built (before transpilation)
- `09_circuit_diagram_transpiled.txt` — the same circuit in {CNOT, u3}; long, kept for gate-by-gate checking
- `10_readout.txt` — summed qubits, one-shot probabilities and the reduced density matrix
- `11_readout.json` — one-shot probabilities and both reduced density matrices as numbers

## Checks

| statement | value | reference | passed |
|---|---|---|---|
| M is symmetric | 0 | 0 | yes |
| smallest eigenvalue of M (>= 0 means the whole IF is a contraction) | 0.0144554444268 |  | yes |
| the pieces sum to M | 0 | 0 | yes |
| rescaling read off the circuit equals the LP optimum | 23.7001036153 | 23.7001036153 | yes |
| circuit RDM equals the independent classical path sum | 2.22045266669e-16 | 0 | yes |
| P_ro = d^-2(N-1) rho^2 alpha^-2 for the first population | 7.35552239285e-05 | 7.35552239285e-05 | yes |

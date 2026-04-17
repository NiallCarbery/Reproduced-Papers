import os
import time
import gc
import numpy as np
import multiprocessing
from qutip import sigmax, sigmaz, qeye, tensor, basis, sesolve, fidelity


sigma_x = sigmax()
sigma_z = sigmaz()
identity = qeye(2)


def ising_parameters(N, seed):
    np.random.seed(seed)
    J = np.random.uniform(0, 1, size=(N, N))
    h = np.random.uniform(0, 1, size=N)
    return J, h


def ising_hamiltonian(N, J, h):
    H = 0
    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] != 0:
                op_list = [identity] * N
                op_list[i] = sigma_z
                op_list[j] = sigma_z
                H += J[i, j] * tensor(op_list)
    for i in range(N):
        if h[i] != 0:
            op_list = [identity] * N
            op_list[i] = sigma_z
            H += h[i] * tensor(op_list)
    return H


def transverse_hamiltonian(N):
    """Create transverse field Hamiltonian H_i = -sum_i sigma_x^i
    Standard QA convention: ground state is |+>^N (all spins in +x direction)
    """
    H = 0
    for i in range(N):
        op_list = [identity] * N
        op_list[i] = sigma_x
        H -= tensor(op_list)
    return H


def get_ground_state(H):
    _, vecs = H.eigenstates(eigvals=1)
    return vecs[0]


def calculate_success_probability_constrained_optimized(
    T, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
):
    """Optimized calculation for Constrained protocol.

    The constrained QND-like protocol
    H_int(t) = f(t) * H_f ⊗ X_M  (where X_M = x0 * sigma_z)
    H_total(t) = H_S(t) ⊗ I + f(t) * H_f ⊗ (x0 * sigma_z)
    where H_S(t) = (1-f(t))*H_i + f(t)*H_f and f(t) = t/T
    """
    t_eval = [0, T]

    # System Hamiltonian terms: H_S(t) ⊗ I = (1-t/T)*H_i ⊗ I + (t/T)*H_f ⊗ I
    H_initial_term = tensor(H_i, identity)  # (1-t/T) coefficient
    H_final_term = tensor(H_f, identity)  # (t/T) coefficient

    # Interaction term: f(t) * H_f ⊗ (x0 * sigma_z) = (t/T) * H_f ⊗ (x0 * sigma_z)
    H_interaction = x0 * tensor(H_f, sigma_z)  # (t/T) coefficient

    H_list = [
        [H_initial_term, "1.0 - t / T"],
        [H_final_term, "t / T"],
        [H_interaction, "t / T"],
    ]

    psi0 = tensor(sys_initial_ground, basis(2, 0))
    args = {"T": T}
    nsteps = max(100000, int(200000 * (T / 5.0)))
    options = {"nsteps": nsteps}

    result = sesolve(H_list, psi0, t_eval, args=args, options=options)
    psi_final = result.states[-1]

    sys_rho_final = psi_final.ptrace(list(range(N)))
    fid = fidelity(sys_rho_final, sys_target_rho) ** 2

    del result, psi_final, sys_rho_final, H_initial_term, H_final_term, H_interaction
    return fid


def find_T_for_target_probability(
    H_i,
    H_f,
    sys_initial_ground,
    sys_target_rho,
    N,
    target_p=0.5,
    T_guess=5.0,
    max_iter=10,
    tol=0.005,
):
    """Find annealing time using iterative LZ extrapolation to achieve target probability.

    Uses the LZ formula iteratively to refine T until the coherent protocol (x0=0)
    achieves approximately target_p success probability.
    """
    T_current = T_guess

    for iteration in range(max_iter):
        # Calculate fidelity with current T at x0=0 (coherent baseline)
        p_current = calculate_success_probability_constrained_optimized(
            T_current, H_i, H_f, 0.0, sys_initial_ground, sys_target_rho, N
        )

        # Handle edge cases
        if p_current >= 0.99:
            T_current = T_current * 0.5
            continue
        if p_current <= 0.01:
            T_current = T_current * 2.0
            continue

        # Check convergence
        if abs(p_current - target_p) < tol:
            break

        # LZ extrapolation: T_new = T_current * log(1 - target_p) / log(1 - p_current)
        T_next = T_current * np.log(1 - target_p) / np.log(1 - p_current)
        T_current = max(0.1, T_next)
        print(T_current)

    return T_current


def _single_realization_worker_wrapper(args):
    return _single_realization_worker_scan(*args)


def _single_realization_worker_scan(realization, N, x0_values):
    try:
        J_matrix, h_vector = ising_parameters(N, seed=realization)
        H_f = ising_hamiltonian(N, J_matrix, h_vector)
        H_i = transverse_hamiltonian(N)

        sys_initial_ground = get_ground_state(H_i)
        sys_target_ground = get_ground_state(H_f)
        sys_target_rho = sys_target_ground * sys_target_ground.dag()

        # Find T_ext for coherent protocol (x0=0)
        T_ext = find_T_for_target_probability(
            H_i, H_f, sys_initial_ground, sys_target_rho, N, target_p=0.5
        )

        fidelities = []
        for x0 in x0_values:
            fid = calculate_success_probability_constrained_optimized(
                T_ext, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
            )
            fidelities.append(fid)

        gc.collect()
        return fidelities
    except Exception as e:
        print(f"Error in realization {realization}: {e}")
        return []


def calculate_fidelity_vs_x0(N, x0_values, n_instances, max_workers):
    print(
        f"Computing N={N} across {len(x0_values)} x0 values with {n_instances} instances..."
    )

    tasks = [(i, N, x0_values) for i in range(n_instances)]
    all_fidelities = []

    # maxtasksperchild: recycle each worker after this many tasks to bound memory
    # growth, but keep it >1 to avoid 100 process-restart overheads on 32 CPUs.
    # chunksize=1 keeps imap_unordered responsive so fast tasks don't stall
    # behind slow ones.
    recycle_after = max(1, n_instances // (max_workers * 2))
    with multiprocessing.Pool(
        processes=max_workers,
        maxtasksperchild=recycle_after,
    ) as pool:
        for i, fids in enumerate(
            pool.imap_unordered(_single_realization_worker_wrapper, tasks, chunksize=1)
        ):
            if fids:
                all_fidelities.append(fids)
                print(f"N={N}: {i}/{n_instances} done")

    if not all_fidelities:
        return np.zeros(len(x0_values))

    avg_fidelities = np.mean(all_fidelities, axis=0)
    print(f"N={N}: Completed {len(all_fidelities)} instances")
    return avg_fidelities


if __name__ == "__main__":
    x0_values = np.linspace(0, 4, 15)
    n_instances = 100

    import multiprocessing

    available_cores = multiprocessing.cpu_count()
    # Use all physical cores; cap at n_instances since extra workers are idle.
    max_workers = min(available_cores, n_instances)

    N_values_even = [4, 6, 8, 10]
    N_values_odd = [3, 5, 7, 9, 11]

    print(f"Generating Figure 5(b,c) OPTIMIZED data...")

    fidelities_even = {}
    for N in N_values_even:
        current_workers = max_workers

        fidelities_even[N] = calculate_fidelity_vs_x0(
            N, x0_values, n_instances, current_workers
        )
        # Save progress after completing this N (even)
        os.makedirs("output", exist_ok=True)
        np.savez(
            "output/figure5bc_constrained.npz",
            x0_values=x0_values,
            N_values_even=N_values_even,
            N_values_odd=N_values_odd,
            fidelities_even=fidelities_even,
            fidelities_odd={},
            n_instances=n_instances,
            last_completed_N=N,
            phase="even",
            timestamp=time.time(),
        )
        print(f"Saved progress after N={N} (even)")

    fidelities_odd = {}
    for N in N_values_odd:
        current_workers = max_workers

        fidelities_odd[N] = calculate_fidelity_vs_x0(
            N, x0_values, n_instances, current_workers
        )

        # Save progress after completing this N (odd)
        os.makedirs("output", exist_ok=True)
        np.savez(
            "output/figure5bc_constrained.npz",
            x0_values=x0_values,
            N_values_even=N_values_even,
            N_values_odd=N_values_odd,
            fidelities_even=fidelities_even,
            fidelities_odd=fidelities_odd,
            n_instances=n_instances,
            last_completed_N=N,
            phase="odd",
            timestamp=time.time(),
        )
        print(f"Saved progress after N={N} (odd)")

    os.makedirs("output", exist_ok=True)
    np.savez(
        "output/figure_5bc.npz",
        x0_values=x0_values,
        N_values_even=N_values_even,
        N_values_odd=N_values_odd,
        fidelities_even=fidelities_even,
        fidelities_odd=fidelities_odd,
        n_instances=n_instances,
    )
    print("Save")

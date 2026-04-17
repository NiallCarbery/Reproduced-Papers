import os
import gc
import argparse
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
    H = 0
    for i in range(N):
        op_list = [identity] * N
        op_list[i] = sigma_x
        H -= tensor(op_list)
    return H


def get_ground_state(H):
    _, vecs = H.eigenstates(eigvals=1)
    return vecs[0]


def calculate_success_probability_optimized(
    T, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
):
    """Optimized calculation for QND protocol"""
    t_eval = [0, T]

    # Pre-calculate operators
    meter_op = identity + x0 * sigma_z
    H_decay = tensor(H_i, meter_op)
    H_rise = tensor(H_f, meter_op)

    # H(t) = (1-t/T)*H_decay + (t/T)*H_rise
    H_list = [[H_decay, "1.0 - t / T"], [H_rise, "t / T"]]

    psi0 = tensor(sys_initial_ground, basis(2, 0))
    args = {"T": T}
    options = {"nsteps": 100000}

    result = sesolve(H_list, psi0, t_eval, args=args, options=options)
    psi_final = result.states[-1]

    rho_final = psi_final.ptrace(list(range(N)))
    fid = fidelity(rho_final, sys_target_rho) ** 2

    del result, psi_final, rho_final, H_decay, H_rise
    return fid


def calculate_success_probability_constrained_optimized(
    T, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
):
    """Optimized calculation for Constrained protocol"""
    t_eval = [0, T]

    # H_sys(t) = (1-t/T)*H_i + (t/T)*H_f
    # H_int(t) = (t/T)*H_f tensor (x0*sigma_z)
    # Total = (1-t/T)*(H_i tensor I) + (t/T)*(H_f tensor I) + (t/T)*(H_f tensor x0*sigma_z)
    #       = (1-t/T)*(H_i tensor I) + (t/T)*(H_f tensor (I + x0*sigma_z))

    meter_op_constrained = identity + x0 * sigma_z

    H_decay = tensor(H_i, identity)
    H_rise = tensor(H_f, meter_op_constrained)

    H_list = [[H_decay, "1.0 - t / T"], [H_rise, "t / T"]]

    psi0 = tensor(sys_initial_ground, basis(2, 0))
    args = {"T": T}
    options = {"nsteps": 100000}

    result = sesolve(H_list, psi0, t_eval, args=args, options=options)
    psi_final = result.states[-1]

    rho_final = psi_final.ptrace(list(range(N)))
    fid = fidelity(rho_final, sys_target_rho) ** 2

    del result, psi_final, rho_final, H_decay, H_rise
    return fid


def find_T_for_target_probability(
    H_i, H_f, x0, sys_initial_ground, sys_target_rho, N, target_p=0.5, T_guess=5.0
):
    """Find annealing time that gives target success probability using Landau-Zener extrapolation"""
    p_guess = calculate_success_probability_optimized(
        T_guess, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
    )

    if p_guess >= 1.0:
        return 0.1
    if p_guess <= 0.0:
        return T_guess * 2  # Try larger T if p is 0

    T_ext = T_guess * np.log(1 - target_p) / np.log(1 - p_guess)
    return max(0.1, T_ext)


def _single_realization_worker_wrapper(args):
    return _single_realization_worker(*args)


def _single_realization_worker(realization, N, x0, nT, constrained_only=False):
    try:
        J_matrix, h_vector = ising_parameters(N, seed=realization)
        H_f = ising_hamiltonian(N, J_matrix, h_vector)
        H_i = transverse_hamiltonian(N)

        # Pre-calculate ground states ONCE per realization
        sys_initial_ground = get_ground_state(H_i)
        sys_target_ground = get_ground_state(H_f)
        sys_target_rho = sys_target_ground * sys_target_ground.dag()

        # Find T_ext using coherent protocol (x0=0)
        T_ext_coh = find_T_for_target_probability(
            H_i, H_f, 0.0, sys_initial_ground, sys_target_rho, N, target_p=0.5
        )

        # Generate T values
        T_values = np.logspace(np.log10(0.1 * T_ext_coh), np.log10(10 * T_ext_coh), nT)

        p_qnd = []
        p_coh = []
        p_con = []

        for T in T_values:
            if not constrained_only:
                p_qnd.append(
                    calculate_success_probability_optimized(
                        T, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
                    )
                )
            p_coh.append(
                calculate_success_probability_optimized(
                    T, H_i, H_f, 0.0, sys_initial_ground, sys_target_rho, N
                )
            )
            p_con.append(
                calculate_success_probability_constrained_optimized(
                    T, H_i, H_f, x0, sys_initial_ground, sys_target_rho, N
                )
            )

        # Calculate TTS
        def get_tts(probs):
            tts_vals = []
            for T, p in zip(T_values, probs):
                if p >= 0.999:
                    p = 0.999
                tts = T * np.log(0.05) / np.log(1 - p)
                tts_vals.append(tts)
            return min(tts_vals)

        tts_qnd = get_tts(p_qnd) if not constrained_only else np.nan
        tts_coh = get_tts(p_coh)
        tts_con = get_tts(p_con)

        ratio_qnd = tts_qnd / tts_coh if (not constrained_only and tts_coh > 0) else np.nan
        ratio_con = tts_con / tts_coh if tts_coh > 0 else np.nan

        gc.collect()
        return ratio_qnd, ratio_con
    except Exception as e:
        print(f"Error in realization {realization}: {e}")
        return np.nan, np.nan


def calculate_tts_ratio_for_N(N, n_instances, x0, nT, max_workers, constrained_only=False):
    print(f"Computing N={N} with {n_instances} instances...")

    tasks = [(i, N, x0, nT, constrained_only) for i in range(n_instances)]
    ratios_qnd = []
    ratios_con = []

    with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        for i, (rq, rc) in enumerate(
            pool.imap_unordered(_single_realization_worker_wrapper, tasks)
        ):
            if not constrained_only and not np.isnan(rq):
                ratios_qnd.append(rq)
            if not np.isnan(rc):
                ratios_con.append(rc)
            if i % 10 == 0:
                print(f"N={N}: {i}/{n_instances} done")

    avg_qnd = np.mean(ratios_qnd) if ratios_qnd else np.nan
    avg_con = np.mean(ratios_con) if ratios_con else np.nan

    if constrained_only:
        print(f"N={N}: Avg Con={avg_con:.4f}")
    else:
        print(f"N={N}: Avg QND={avg_qnd:.4f}, Avg Con={avg_con:.4f}")
    return avg_qnd, avg_con


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 5(a) data with optional constrained-only mode."
    )
    parser.add_argument(
        "--constrained-only",
        action="store_true",
        help="Only compute constrained protocol ratios (skip unconstrained/QND protocol).",
    )
    args = parser.parse_args()

    N_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    x0 = 1.0
    n_instances = 100
    nT = 10

    import multiprocessing

    available_cores = multiprocessing.cpu_count()
    max_workers = min(10, n_instances)

    if args.constrained_only:
        print("Generating Figure 5(a) OPTIMIZED data (constrained-only mode)...")
    else:
        print("Generating Figure 5(a) OPTIMIZED data...")

    tts_ratios_qnd = []
    tts_ratios_con = []

    for N in N_values:
        # Adjust workers for larger N to save memory
        current_workers = max_workers
        if N >= 8:
            current_workers = max(1, current_workers // 2)

        if N >= 10:
            current_workers = max(1, current_workers // 8)

        rq, rc = calculate_tts_ratio_for_N(
            N, n_instances, x0, nT, current_workers, args.constrained_only
        )
        tts_ratios_qnd.append(rq)
        tts_ratios_con.append(rc)

    os.makedirs("output", exist_ok=True)
    np.savez(
        "output/x01_figure_5a_data_optimized.npz",
        N_values=N_values,
        tts_ratios_qnd=tts_ratios_qnd,
        tts_ratios_con=tts_ratios_con,
        x0=x0,
        n_instances=n_instances,
    )

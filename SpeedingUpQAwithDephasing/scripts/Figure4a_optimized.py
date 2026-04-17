import os
import time
import gc
import numpy as np
import multiprocessing
from qutip import sigmax, sigmaz, qeye, tensor, basis, sesolve, fidelity

sigma_x = sigmax()
sigma_z = sigmaz()
identity = qeye(2)


def landau_zener_hamiltonian(t, v, g):
    return (v * t / 2) * sigma_z + (g / 2) * sigma_x


def get_instantaneous_eigenstates(t, v, g):
    H = landau_zener_hamiltonian(t, v, g)
    return H.eigenstates()


def _run_worker_wrapper(args):
    return _run_worker(*args)


def _run_worker(i, T, omega_values, g, x0):
    try:
        v = 20.0 / T
        t_start = -10.0 / v
        t_end = 10.0 / v

        # Pre-calculate eigenstates
        _, init_eigenstates = get_instantaneous_eigenstates(t_start, v, g)
        sys_initial = init_eigenstates[0]
        _, eigenstates = get_instantaneous_eigenstates(t_end, v, g)
        sys_target = eigenstates[0]

        sys_target_rho = sys_target * sys_target.dag()

        # Initial state
        meter_initial = basis(2, 0)
        psi0 = tensor(sys_initial, meter_initial)

        N = len(sys_initial.dims[0])

        # Hamiltonian Construction
        # H_LZ(t) = (v/2 * sigma_z) * t + (g/2 * sigma_x)
        # H_sys = H_LZ tensor I
        # H_int = x0 * (H_LZ tensor sigma_z)
        # H_meter = omega * (I tensor sigma_x)

        # Combine H_sys and H_int:
        # H_sys + H_int = H_LZ tensor (I + x0 * sigma_z)
        # Let M = I + x0 * sigma_z
        # Term 1: (v/2 * sigma_z * t) tensor M = (v/2 * sigma_z tensor M) * t
        # Term 2: (g/2 * sigma_x) tensor M

        meter_op = identity + x0 * sigma_z

        Hz_sys = (v / 2) * sigma_z
        Hx_sys = (g / 2) * sigma_x

        H_t_term = tensor(Hz_sys, meter_op)
        H_const_term = tensor(Hx_sys, meter_op)
        H_meter_base = tensor(identity, sigma_x)

        t_eval = [t_start, t_end]

        row = np.zeros(len(omega_values))

        # 1. Baseline (omega=0)
        H_baseline = [[H_t_term, "t"], H_const_term]

        result_base = sesolve(H_baseline, psi0, t_eval)
        psi_final_base = result_base.states[-1]
        rho_final_base = psi_final_base.ptrace(list(range(N)))
        baseline_fid = fidelity(rho_final_base, sys_target_rho) ** 2

        del result_base, psi_final_base, rho_final_base

        # 2. Loop omega
        for j, omega in enumerate(omega_values):
            # H = H_baseline + omega * H_meter_base
            H = [[H_t_term, "t"], H_const_term + omega * H_meter_base]

            result = sesolve(H, psi0, t_eval)
            psi_final = result.states[-1]
            rho_final = psi_final.ptrace(list(range(N)))
            fid = fidelity(rho_final, sys_target_rho) ** 2

            row[j] = fid - baseline_fid

            del result, psi_final, rho_final

        gc.collect()
        return i, row
    except Exception as e:
        print(f"Error in T index {i}: {e}")
        return i, np.zeros(len(omega_values))


def parallel_fidelity_difference(T_values, omega_values, g, max_workers):
    x0 = 1
    fidelity_diff = np.zeros((len(T_values), len(omega_values)))

    tasks = [(i, T, omega_values, g, x0) for i, T in enumerate(T_values)]

    start = time.time()
    with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        for i, row in pool.imap_unordered(_run_worker_wrapper, tasks):
            fidelity_diff[i, :] = row
            if i % 10 == 0:
                print(f"Completed T index {i}/{len(T_values)}")

    elapsed = time.time() - start
    print(f"Parallel run finished in {elapsed:.1f}s using {max_workers} workers")
    return fidelity_diff


if __name__ == "__main__":
    T_values = np.linspace(0.0, 10, 100)
    omega_values = np.linspace(0.0, 10, 100)
    g = 1.0

    import multiprocessing

    available_cores = multiprocessing.cpu_count()
    max_workers = min(max(1, int(available_cores * 0.75)), len(T_values))

    os.makedirs("output", exist_ok=True)
    print(
        f"Running OPTIMIZED with {max_workers} workers, len(T)={len(T_values)}, len(omega)={len(omega_values)}"
    )

    fidelity_diff = parallel_fidelity_difference(T_values, omega_values, g, max_workers)

    outpath = "output/Fig4a_optimized.npz"
    np.savez_compressed(
        outpath, T=T_values, omega=omega_values, fidelity_diff=fidelity_diff, g=g
    )
    print(f"Saved results to {outpath}")

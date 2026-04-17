import os
# Prevent each worker process from spawning multiple BLAS/OMP threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
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
    H = 0
    for i in range(N):
        op_list = [identity] * N
        op_list[i] = sigma_x
        H -= tensor(op_list)
    return H


def get_ground_state(H):
    _, vecs = H.eigenstates(eigvals=1)
    return vecs[0]


def _run_worker_wrapper(args):
    return _run_worker(*args)


def _worker_initializer():
    """Called once when each pool worker starts. Ensures single-threaded BLAS."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _run_worker(run, T_values, omega_values, N):
    """Worker for one random run (returns accumulated fidelity-diff matrix)"""
    try:
        # Generate parameters
        J_matrix, h_vector = ising_parameters(N, seed=run)

        # Construct Hamiltonians
        H_f = ising_hamiltonian(N, J_matrix, h_vector)
        H_i = transverse_hamiltonian(N)

        # Pre-calculate ground states
        sys_initial_ground = get_ground_state(H_i)
        sys_target_ground = get_ground_state(H_f)

        # Target density matrix
        sys_target_rho = sys_target_ground * sys_target_ground.dag()

        # Initial state
        psi0 = tensor(sys_initial_ground, basis(2, 0))

        # Pre-calculate Hamiltonian terms
        x0 = 1.0
        meter_op = identity + x0 * sigma_z

        H_decay = tensor(H_i, meter_op)
        H_rise = tensor(H_f, meter_op)
        H_const_base = tensor(tensor([identity] * N), sigma_x)

        fidelity_diff_acc = np.zeros((len(T_values), len(omega_values)))

        coeff_decay = "1.0 - t / T"
        coeff_rise = "t / T"

        for i, T in enumerate(T_values):
            t_eval = [0, T]
            args = {"T": T}

            # 1. Calculate Baseline (omega = 0)
            H_baseline = [[H_decay, coeff_decay], [H_rise, coeff_rise]]

            result_baseline = sesolve(H_baseline, psi0, t_eval, args=args)
            psi_final_baseline = result_baseline.states[-1]

            rho_final_baseline = psi_final_baseline.ptrace(list(range(N)))
            fid_baseline = fidelity(rho_final_baseline, sys_target_rho) ** 2

            # Explicit cleanup
            del result_baseline, psi_final_baseline

            for j, omega in enumerate(omega_values):
                H_total = [
                    omega * H_const_base,
                    [H_decay, coeff_decay],
                    [H_rise, coeff_rise],
                ]

                result = sesolve(H_total, psi0, t_eval, args=args)
                psi_final = result.states[-1]

                rho_final = psi_final.ptrace(list(range(N)))
                fid = fidelity(rho_final, sys_target_rho) ** 2

                fidelity_diff_acc[i, j] += fid - fid_baseline

                # Explicit cleanup inside inner loop
                del result, psi_final, rho_final

        # Final cleanup before returning
        del H_decay, H_rise, H_const_base, H_i, H_f, psi0
        gc.collect()

        return fidelity_diff_acc
    except Exception as e:
        print(f"Error in run {run}: {e}")
        return np.zeros((len(T_values), len(omega_values)))


def parallel_fidelity_matrix(T_values, omega_values, N, num_runs, max_workers):
    """Parallel launcher with time tracking"""
    fidelity_diff_accumulated = np.zeros((len(T_values), len(omega_values)))

    start = time.time()

    # Prepare arguments for map
    tasks = [(r, T_values, omega_values, N) for r in range(num_runs)]

    # Use a 'spawn' context for Pool to avoid leaked semaphore objects
    # (resource_tracker warnings on shutdown). Also use maxtasksperchild=1
    # to restart workers and return memory to the OS.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=max_workers,
        maxtasksperchild=1,
        initializer=_worker_initializer,
    ) as pool:
        results = pool.imap_unordered(_run_worker_wrapper, tasks)

        for i, res in enumerate(results):
            fidelity_diff_accumulated += res
            print(f"Run {i + 1}/{num_runs} completed.")

    elapsed = time.time() - start
    print(f"Completed {num_runs} runs in {elapsed:.1f}s using {max_workers} workers")
    return fidelity_diff_accumulated / num_runs


if __name__ == "__main__":
    # Parameters
    T_values = np.linspace(0.0001, 10, 100)
    omega_values = np.linspace(0, 10, 100)
    N = 6
    num_runs = 100
    max_workers = min(10, num_runs)

    os.makedirs("output", exist_ok=True)

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    print(
        f"Running OPTIMIZED simulation with {max_workers} workers, {num_runs} runs, N={N}"
    )

    fidelity_ising_avg = parallel_fidelity_matrix(
        T_values, omega_values, N, num_runs, max_workers
    )

    outpath = "output/6-Fig4b.npz"
    np.savez_compressed(
        outpath,
        T=T_values,
        omega=omega_values,
        fidelity_ising_avg=fidelity_ising_avg,
        N=N,
        num_runs=num_runs,
    )
    print(f"Saved results to {outpath}")

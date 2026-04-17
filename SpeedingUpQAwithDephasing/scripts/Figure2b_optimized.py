import os
import time
import gc
import argparse
import sys
import multiprocessing

# Reduce multithreading inside BLAS/LAPACK libraries to avoid oversubscription
# Set before importing numpy or other numeric libraries
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
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


def _run_worker(run, T_values, x0_values, N):
    try:
        print(f"Worker started for run {run}", flush=True)
        J_matrix, h_vector = ising_parameters(N, seed=run)
        H_f = ising_hamiltonian(N, J_matrix, h_vector)
        H_i = transverse_hamiltonian(N)

        # Pre-calculate ground states ONCE per run
        sys_initial_ground = get_ground_state(H_i)
        sys_target_ground = get_ground_state(H_f)

        sys_target_rho = sys_target_ground * sys_target_ground.dag()
        psi0 = tensor(sys_initial_ground, basis(2, 0))

        fidel_acc = np.zeros((len(T_values), len(x0_values)))

        coeff_decay = "1.0 - t / T"
        coeff_rise = "t / T"

        # Swapped loops to optimize H construction (reuse H for all T per x0)
        # Also added progress logging
        for j, x0 in enumerate(x0_values):
            if j % 10 == 0:
                print(
                    f"Run {run}: Processing x0 index {j}/{len(x0_values)}", flush=True
                )

            # H_total = H(t) tensor (I + x0 * sigma_z)
            # M = I + x0 * sigma_z
            meter_op = identity + x0 * sigma_z

            # H_total = (1-t/T) * (H_i tensor M) + (t/T) * (H_f tensor M)
            H_decay = tensor(H_i, meter_op)
            H_rise = tensor(H_f, meter_op)

            H_list = [[H_decay, coeff_decay], [H_rise, coeff_rise]]

            for i, T in enumerate(T_values):
                t_eval = [0, T]
                args = {"T": T}

                result = sesolve(
                    H_list, psi0, t_eval, args=args, options={"nsteps": 100000}
                )
                psi_final = result.states[-1]

                rho_final = psi_final.ptrace(list(range(N)))
                fid = fidelity(rho_final, sys_target_rho) ** 2
                fidel_acc[i, j] += fid

                del result, psi_final, rho_final

            del H_decay, H_rise

        gc.collect()
        return fidel_acc
    except Exception as e:
        print(f"Error in run {run}: {e}")
        return np.zeros((len(T_values), len(x0_values)))


def simulate_ising_qnd_fidelity_parallel(T_values, x0_values, N, num_runs, max_workers, runs_per_task=1, maxtasksperchild=1):
    accumulated = np.zeros((len(T_values), len(x0_values)))
    start = time.time()

    # Optionally group multiple runs into a single task to reduce scheduling overhead
    tasks = []
    if runs_per_task <= 1:
        tasks = [(r, T_values, x0_values, N) for r in range(num_runs)]
    else:
        for start_run in range(0, num_runs, runs_per_task):
            # a task will be identified by its starting run index and handle up to runs_per_task runs
            tasks.append((start_run, runs_per_task, T_values, x0_values, N))

    with multiprocessing.Pool(processes=max_workers, maxtasksperchild=maxtasksperchild) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_worker_wrapper, tasks)):
            accumulated += res
            print(f"Run task {i+1}/{len(tasks)} done")

    elapsed = time.time() - start
    print(f"Completed {num_runs} runs in {elapsed:.1f}s using {max_workers} workers")
    return accumulated / float(num_runs)


if __name__ == "__main__":
    T_values = np.linspace(0.0001, 10, 100)
    x0_values = np.linspace(0, 10, 100)
    N = 6
    num_runs = 100

    import multiprocessing

    available_cores = multiprocessing.cpu_count()
    max_workers = min(10, num_runs)

    os.makedirs("output", exist_ok=True)
    print(
        f"Starting OPTIMIZED parallel run: N={N}, runs={num_runs}, workers={max_workers}"
    )
    fidelities = simulate_ising_qnd_fidelity_parallel(
        T_values, x0_values, N, num_runs, max_workers
    )

    outpath = "output/6-mesh2b.npz"
    np.savez_compressed(
        outpath,
        T=T_values,
        x0=x0_values,
        fidelities=fidelities,
        N=N,
        num_runs=num_runs,
    )
    print(f"Saved {outpath}")

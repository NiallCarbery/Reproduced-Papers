import os
import time
import gc
import numpy as np
from qutip import sigmax, sigmaz, qeye, sesolve
from multiprocessing import Pool

# Pauli matrices
sigma_x = sigmax()
sigma_z = sigmaz()
identity = qeye(2)


def landau_zener_hamiltonian(t, v, g):
    return (v * t / 2) * sigma_z + (g / 2) * sigma_x


def get_instantaneous_eigenstates(t, v, g):
    H = landau_zener_hamiltonian(t, v, g)
    return H.eigenstates()


def _compute_fidelity_row(args):
    """Helper for parallel computation: compute fidelity row for a single T."""
    try:
        T, x0_values, g = args
        v = 20.0 / T
        t_start = -10.0 / v
        t_end = 10.0 / v

        # Pre-calculate eigenstates
        _, eigenstates = get_instantaneous_eigenstates(t_end, v, g)
        target_state = eigenstates[0]

        _, init_eigenstates = get_instantaneous_eigenstates(t_start, v, g)
        initial_state = init_eigenstates[0]

        row = np.zeros(len(x0_values))

        # Pre-calculate Hamiltonian terms
        # H(t) = (v/2 * sigma_z) * t + (g/2 * sigma_x)
        # H_plus(t) = (1+x0) * H(t)

        Hz = (v / 2) * sigma_z
        Hx = (g / 2) * sigma_x

        t_eval = [t_start, t_end]

        for j, x0 in enumerate(x0_values):
            scale = 1.0 + x0

            # H = [ [scale * Hz, "t"], scale * Hx ]
            H = [[scale * Hz, "t"], scale * Hx]

            result = sesolve(H, initial_state, t_eval)
            final_state_plus = result.states[-1]

            fidelity = abs(target_state.overlap(final_state_plus)) ** 2
            row[j] = fidelity

            del result, final_state_plus

        gc.collect()
        return row
    except Exception as e:
        print(f"Error processing T={T}: {e}")
        return np.zeros(len(x0_values))


def simulate_qnd_protocol_fidelity_parallel(T_values, x0_values, g, nprocs):
    """
    Parallel compute fidelity.
    """
    args_iter = [(T, x0_values, g) for T in T_values]

    fidelities = np.zeros((len(T_values), len(x0_values)))

    start = time.time()

    with Pool(processes=nprocs, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap(_compute_fidelity_row, args_iter)):
            fidelities[i, :] = row
            if i % 10 == 0:
                print(f"Completed {i}/{len(T_values)}")

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s using {nprocs} workers")
    return fidelities


if __name__ == "__main__":
    # Parameters
    T_values = np.linspace(0, 10, 100)
    x0_values = np.linspace(0, 1, 100)
    g = 1.0

    import multiprocessing

    available_cores = multiprocessing.cpu_count()
    max_workers = min(max(1, int(available_cores * 0.75)), len(T_values))

    # Use the directory one level above this script for output files
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(parent_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Running OPTIMIZED with {max_workers} workers, len(T)={len(T_values)}, len(x0)={len(x0_values)}"
    )

    fidelities = simulate_qnd_protocol_fidelity_parallel(
        T_values, x0_values, g, max_workers
    )

    outpath = os.path.join(output_dir, "mesh2a_optimized.npz")
    np.savez_compressed(outpath, T=T_values, x0=x0_values, fidelities=fidelities, g=g)
    print(f"Saved results to {outpath}")

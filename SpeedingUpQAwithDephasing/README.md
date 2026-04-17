# Speeding Up Quantum Annealing with Engineered Dephasing (Reproduction)

This folder contains a reproduction project for the paper:

- **Title:** Speeding Up Quantum Annealing with Engineered Dephasing
- **Original Paper URL:** https://quantum-journal.org/papers/q-2025-05-05-1731/
- **Type:** Independent reproduction for learning and verification

## Disclaimer

This is an unofficial reproduction project.

I am not affiliated with the original paper authors, their institutions, or the publisher.

## Folder Structure

- `DephasingPaper.ipynb`: Main notebook used for exploration and reproduction steps.
- `enviroment.yml`: Environment specification used for dependencies.
- `data/`: Saved data arrays (`.npz`) used to regenerate figures.
- `scripts/`: Optimized scripts for reproducing individual figures.

## Included Figure Scripts

- `scripts/Figure2a_optimized.py`
- `scripts/Figure2b_optimized.py`
- `scripts/Figure4a_optimized.py`
- `scripts/Figure4b_optimized.py`
- `scripts/Figure5a_optimized.py`
- `scripts/Figure5bc_optimized.py`

## How To Use

1. Create and activate the environment from `enviroment.yml`.
2. Run the notebook `DephasingPaper.ipynb` for end-to-end reproduction.
3. Run individual scripts in `scripts/` to reproduce specific figures.

## Notes

- Numerical results may vary slightly across hardware and library versions.
- Data files in `data/` are included to support reproducibility and faster reruns.

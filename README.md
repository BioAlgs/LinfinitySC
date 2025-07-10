# Robust Counterfactual and Synthetic Controls under Dependent Data

This repository contains simulation code and real-data applications for evaluating treatment effects using various regularized synthetic control methods, including novel approaches based on the L-infinity norm.

## Key Features

- Implements multiple estimators: Synthetic Control, Lasso, Ridge, Elastic Net, L-infinity, and L1 + L-infinity.
- Supports both simulation studies and empirical applications.
- Methods accommodate dependent and high-dimensional data structures.
- Plots and evaluates treatment effect estimates using cross-validation or holdout validation.

## Getting Started

### Running Simulations

To run the simulation study, execute:

```bash
python run_sim.py
```

This script runs simulations across multiple methods and outputs results for comparison.

### Real Data Applications

The project contains real-data scripts such as:

- `sho0.py`: Analyze the Technology sector under short-selling constraints.
- `sho0_avg.py`: Computes average treatment effects across companies within a sector.
- `paper_ex.py`: Additional examples included in the paper.
- `sim.ipynb`, `solution_path.ipynb`: Jupyter notebooks for simulation and solution path visualization.
- `exam.ipynb`: Walkthrough for methodological illustration.

You may need to modify paths to load appropriate data (e.g., `SHO/*.csv` files).


## License

MIT License

---
For questions or contributions, please open an issue or contact the maintainer.

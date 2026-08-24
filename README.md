# NIVS: Neural Information-guided View Selection

NIVS (Neural Information-guided View Selection) is an active view selection framework for Neural Radiance Fields (NeRF). It leverages information-theoretic metrics—such as ray transmittance entropy and degeneracy detection—to select the most informative candidate camera viewpoints during NeRF training.

## Overview

- **`nivsalgorithm.py`**: Implementation of active view selection using ray transmittance entropy and information gain across probe rays.
- **`entropyDegeneracy_function.py`**: Detection of entropy degeneracy and stabilization points over training iterations.
- **`nerf.py`**: PyTorch NeRF architecture with coarse and fine MLPs, positional encoding, and ray sampling.
- **`UtilityNeRF.py`**: Helper utilities for volumetric rendering, query point sampling, and iteration steps.
- **`NIVS_helper_function.py`**: Supporting evaluation and plotting routines.
- **`main-exp1.ipynb`**: Experiment notebook demonstrating training, view selection, and metric evaluation.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter](https://jupyter.org/) (optional, for running notebooks)

Install dependencies using `pip`:

```bash
pip install torch numpy scipy matplotlib jupyter
```

## Getting Started

1. **Dataset**: Place your NeRF dataset (e.g., `tiny_nerf_data.npz` or custom camera poses/images) in the project directory.
2. **Run Experiments**: Open and run [`main-exp1.ipynb`](main-exp1.ipynb) to train the NeRF model with active view selection and visualize transmittance distributions.

## License

This project is licensed under the [MIT License](LICENSE).

## Hi there.
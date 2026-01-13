# CSE6250_SP2025_Project

**Course:** CSE 6250 - [Big Data for Healthcare (BD4H)] - Spring 2025

## Overview

This repository contains Python code implemented in Jupyter Notebooks for exploring and applying the **LLSPIN** (Locally Linear Sparse Prediction) and **LSPIN** algorithms, primarily for feature selection and prediction tasks. The notebooks demonstrate the application of these models to both synthetic datasets and the MNIST handwritten digit dataset. The project also investigates hyperparameter optimization using Optuna.

### What is LLSPIN/LSPIN?

LLSPIN (Locally Linear Sparse Prediction) and LSPIN are neural network architectures designed for interpretable feature selection. They consist of:
- **Gating Network**: Learns which features are relevant for each sample using a hard sigmoid activation
- **Prediction Network**: Makes predictions based on the gated (selected) features
- **Feature Selection**: The gating mechanism allows the model to automatically identify and use only relevant features, making it interpretable and potentially more robust

## Project Structure

```
.
├── README.md                 # This file
├── environment.yml           # Conda environment configuration
├── demo0.ipynb              # Basic LLSPIN implementation on synthetic data
├── demo1.ipynb              # LLSPIN with Optuna hyperparameter optimization (5-feature dataset)
├── demo2.ipynb              # LLSPIN with Optuna on E5 nonlinear synthetic dataset
└── mnist_Lspin_gpu.ipynb    # LSPIN applied to MNIST classification with GPU support
```

## Notebooks

### demo0.ipynb - Basic LLSPIN Implementation

**Purpose:** Demonstrates the core functionality of the LLSPIN model on a simple synthetic dataset.

**Contents:**
- Synthetic data generation (4-feature dataset with two groups)
- Complete LLSPIN implementation:
  - `HardSigmoid`: Custom activation function for gating
  - `GatingNetwork`: Learns feature relevance per sample
  - `PredictionNetwork`: Makes predictions on gated features
  - `LSPIN`: Main model combining gating and prediction networks
- Training loop with validation monitoring
- Evaluation metrics: MSE, R², Accuracy
- Visualization comparing learned gates vs. ground truth feature relevance

**Expected Output:**
- Test MSE: ~0.00005-0.0001
- Test R²: ~0.998+
- Feature gate visualizations showing alignment with ground truth

**Key Focus:** Understanding the core mechanism of LLSPIN for feature selection.

---

### demo1.ipynb - LLSPIN with Optuna Optimization (5-Feature Dataset)

**Purpose:** Expands on `demo0.ipynb` by incorporating hyperparameter optimization using Optuna.

**Contents:**
- LLSPIN implementation (same as demo0)
- Extended synthetic dataset with 5 features
- Optuna objective function tuning:
  - Learning rate (log scale: 1e-2 to 2e-1)
  - Regularization parameter λ (log scale: 1e-3 to 1e-2)
  - Number of training epochs (2000, 5000, 10000, 15000)
- Optuna study with 100 trials
- Best model retraining and evaluation
- Feature gate visualization for optimized model

**Expected Output:**
- Optimized hyperparameters from Optuna study
- Improved model performance compared to default hyperparameters
- Feature selection visualizations

**Key Focus:** Automating hyperparameter tuning to improve LLSPIN performance.

---

### demo2.ipynb - LLSPIN on E5 Nonlinear Dataset

**Purpose:** Demonstrates LLSPIN on a more complex nonlinear synthetic dataset (E5) with Optuna optimization.

**Contents:**
- E5 nonlinear data generation:
  - 21 features (20 + 1 group indicator)
  - Three groups with different feature interactions
  - Nonlinear relationships (multiplicative terms)
- LLSPIN model with LeakyReLU activations
- Optuna optimization focusing on:
  - Learning rate (0.01 to 0.1)
  - Number of epochs (2000, 3000, 5000, 7000)
  - Fixed λ = 1 (as per specification)
- Feature selection analysis (union features, median features per sample)
- Ground truth vs. learned gates visualization
- Prediction vs. ground truth scatter plot

**Expected Output:**
- Test MSE: ~0.0001-0.01 (depending on hyperparameters)
- Feature selection metrics showing sparse feature usage
- Visualizations comparing ground truth and learned feature gates

**Key Focus:** Demonstrating LLSPIN's ability to handle nonlinear relationships and complex feature interactions.

---

### mnist_Lspin_gpu.ipynb - MNIST Classification with LSPIN

**Purpose:** Adapts the LSPIN algorithm for the MNIST handwritten digit classification task.

**Contents:**
- MNIST dataset loading and preprocessing (28×28 = 784 features)
- LSPIN model adapted for classification:
  - CrossEntropyLoss for multi-class classification
  - 10 output classes (digits 0-9)
  - Configurable activation functions (ReLU/Tanh)
- Comprehensive Optuna hyperparameter tuning:
  - Learning rate (log scale: 1e-5 to 1e-1)
  - Number of epochs (10, 20, 30, 50)
  - Prediction network architecture (1-3 layers, 64-512 units)
  - Gating network architecture (1-2 layers, 32-128 units)
  - Hard sigmoid slope `a` (0.5 to 2.0)
  - Noise standard deviation `sigma` (log scale: 0.01 to 1.0)
  - Regularization strength `lam` (log scale: 1e-5 to 1e-2)
  - Activation function selection (ReLU/Tanh)
- Model evaluation:
  - Classification accuracy
  - AUC (One-vs-Rest)
  - Confusion matrix
- Visualization of selected pixels/features in MNIST images (top 30 gates per sample)

**Expected Output:**
- Classification accuracy: varies (typically 5-90% depending on hyperparameters)
- Feature selection visualizations showing which pixels are most important for each digit
- Confusion matrix showing classification performance

**Key Focus:** Applying LSPIN to a real-world image classification problem and demonstrating interpretable feature selection.

**Note:** GPU acceleration is supported but not required. The code will run on CPU if GPU is unavailable.

---

## Dependencies and Environment Setup

### Prerequisites

- **Conda**: If you don't have Conda installed, download and install it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **GPU (Optional)**: CUDA-enabled GPU recommended for `mnist_Lspin_gpu.ipynb` but not required

### Installation Steps

1. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate SP2025P
   ```

3. **Install additional dependencies:**
   
   The notebooks require additional packages not included in `environment.yml`. Install them using pip:
   ```bash
   pip install optuna seaborn jupyter jupyterlab
   ```
   
   Or install individually:
   ```bash
   pip install optuna      # For hyperparameter optimization
   pip install seaborn     # For enhanced visualizations (used in mnist_Lspin_gpu.ipynb)
   pip install jupyter     # For running notebooks
   pip install jupyterlab  # Alternative notebook interface (optional)
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

5. **Run the notebooks:** Open and execute the `.ipynb` files in order (demo0 → demo1 → demo2 → mnist_Lspin_gpu) for a progressive learning experience.

### Environment Details

The `environment.yml` file specifies:

```yaml
name: SP2025P
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.7
  - numpy
  - scipy
  - scikit-learn
  - pandas
  - matplotlib
  - pytest
  - pip
  - pip:
    - tensorflow==1.15.2   # Original author recommended for TensorFlow implementation
    - matplotlib==3.1.2
    - torch==1.13.1
    - torchvision==0.14.1
    - torchaudio==0.13.1
```

**Note:** While the environment specifies Python 3.7, the notebooks have been tested and should work with Python 3.7-3.11. However, for best compatibility, use Python 3.7 as specified.

### Missing Dependencies

The following packages are required but not included in `environment.yml`:
- `optuna` - Hyperparameter optimization framework
- `seaborn` - Statistical data visualization (used in mnist_Lspin_gpu.ipynb)
- `jupyter` or `jupyterlab` - Notebook interface

These should be installed manually after creating the environment (see step 3 above).

---

## Usage Guide

### Running Notebooks Sequentially

For best understanding, run the notebooks in this order:

1. **demo0.ipynb**: Start here to understand the basic LLSPIN architecture
2. **demo1.ipynb**: Learn how to optimize hyperparameters with Optuna
3. **demo2.ipynb**: See LLSPIN applied to more complex nonlinear data
4. **mnist_Lspin_gpu.ipynb**: Apply LSPIN to real-world image classification

### Expected Runtime

- **demo0.ipynb**: ~5-10 minutes (10,000 epochs)
- **demo1.ipynb**: ~2-4 hours (100 Optuna trials, each training for up to 15,000 epochs)
- **demo2.ipynb**: ~1-2 hours (20 Optuna trials, each training for up to 7,000 epochs)
- **mnist_Lspin_gpu.ipynb**: ~3-6 hours (100 Optuna trials, each training for up to 50 epochs)

**Note:** Optuna studies can be interrupted and resumed. Consider reducing `n_trials` for faster experimentation.

### GPU Usage

For `mnist_Lspin_gpu.ipynb`, GPU acceleration will be automatically used if:
- CUDA is available
- PyTorch was installed with CUDA support

To check GPU availability:
```python
import torch
print(torch.cuda.is_available())
```

---

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'optuna'**
   - **Solution**: Install optuna: `pip install optuna`

2. **CUDA/GPU not detected**
   - **Solution**: The code will automatically fall back to CPU. For GPU support, ensure PyTorch was installed with CUDA support matching your CUDA version.

3. **Memory errors when running Optuna studies**
   - **Solution**: Reduce `n_trials` in the Optuna study or reduce batch sizes in data loaders.

4. **Matplotlib display issues in Jupyter**
   - **Solution**: Ensure matplotlib is properly installed and try adding `%matplotlib inline` at the beginning of notebook cells.

5. **Slow execution**
   - **Solution**: 
     - Reduce `n_trials` for Optuna studies
     - Use GPU if available (especially for MNIST)
     - Reduce number of epochs for initial testing

6. **Environment activation issues**
   - **Solution**: Ensure you're using the correct shell (use `conda activate` in PowerShell/Command Prompt, or `source activate` in bash/zsh on Linux/Mac)

---

## Key Concepts

### LLSPIN Components

1. **HardSigmoid**: Custom activation function `clamp(a * x + 0.5, 0, 1)` that creates binary-like gates
2. **GatingNetwork**: Neural network that learns feature relevance scores (α) for each sample
3. **PredictionNetwork**: Standard feedforward network that makes predictions on gated features
4. **Feature Selection**: During training, noise is added to gates (σ) to encourage sparsity; during inference, gates are deterministic

### Hyperparameters

- **`a`**: Hard sigmoid slope (controls gate sharpness)
- **`sigma`**: Noise standard deviation for gating (encourages exploration during training)
- **`lam`**: Regularization strength (weight decay)
- **`lr`**: Learning rate for optimization
- **`pred_hidden`**: Prediction network architecture (list of hidden layer sizes)
- **`gate_hidden`**: Gating network architecture (list of hidden layer sizes)

---

## Results and Interpretations

### Synthetic Datasets (demo0, demo1, demo2)

The notebooks demonstrate that LLSPIN can:
- Identify relevant features that match ground truth
- Handle both linear and nonlinear feature interactions
- Achieve high prediction accuracy while maintaining interpretability

### MNIST (mnist_Lspin_gpu.ipynb)

The MNIST notebook shows:
- How LSPIN can be adapted for classification tasks
- Interpretable feature selection (which pixels matter for each digit)
- The trade-off between model complexity and feature selection sparsity

---

## References

- Original LLSPIN/LSPIN paper (if applicable - add citation here)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Disclaimer

This project was developed for an academic course (CSE 6250) and may not represent production-level code. The implementations are intended for educational purposes and experimentation.

# CSE6250_SP2025_Project

**Course:** CSE 6250 - [Big Data for Healthcare (BD4H)] - Spring 2025

**Overview:**

This repository contains Python code implemented in Jupyter Notebooks for exploring and applying the LLSPIN (Locally Linear Sparse Prediction) and LSPIN algorithms, primarily for feature selection and prediction tasks. The notebooks demonstrate the application of these models to both synthetic datasets and the MNIST handwritten digit dataset. The project also investigates hyperparameter optimization using Optuna.

**Files:**

* **demo0.ipynb:**
    * This notebook demonstrates the basic functionality of the LLSPIN model on a simple synthetic dataset.
    * It includes:
        * Synthetic data generation.
        * Implementation of the LLSPIN components (Hard Sigmoid, Gating Network, Prediction Network).
        * Training and evaluation of the LLSPIN model.
        * Visualization of the learned gates compared to the ground truth feature relevance.
    * Key focus: Understanding the core mechanism of LLSPIN for feature selection.

* **demo1.ipynb:**
    * This notebook expands on `demo0.ipynb` by incorporating hyperparameter optimization using the Optuna library.
    * It includes:
        * The LLSPIN implementation.
        * Definition of an Optuna objective function to tune hyperparameters (e.g., learning rate, network architecture).
        * Running an Optuna study to find the best hyperparameter configuration.
        * Visualization of the feature selection based on the optimized model.
    * Key focus: Automating hyperparameter tuning to improve LLSPIN performance.

* **demo2.ipynb:**
    * This notebook focuses on applying Optuna for hyperparameter tuning in a different experimental setup (likely another synthetic dataset or a modified version of one).
    * It showcases:
        * Defining a model and training loop.
        * Using Optuna to optimize hyperparameters like learning rate and the number of training epochs.
        * Evaluation of the best-performing model based on Mean Squared Error (MSE) and R-squared ($R^2$).
        * Visualization of the predicted values against the ground truth.
    * Key focus: Demonstrating Optuna's versatility in optimizing models for regression tasks.

* **mnist_Lspin_gpu.ipynb:**
    * This notebook adapts the LSPIN algorithms for the MNIST handwritten digit classification task.
    * It leverages GPU acceleration for training (if available).
    * It includes:
        * Loading and preprocessing the MNIST dataset.
        * Implementation of the LSPIN models.
        * Hyperparameter tuning with Optuna.
        * Evaluation of the model's classification accuracy.
        * (Potentially) Visualization of selected pixels/features in MNIST images.
    * Key focus: Applying LSPIN to a real-world image classification problem.

**Dependencies and Environment:**

To ensure the code runs correctly, it is recommended to create a Conda environment using the provided `environment.yml` file.

1.  **Install Conda:** If you don't have Conda installed, download and install it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2.  **Create the environment:** Open a terminal or command prompt and navigate to the directory containing the `environment.yml` file. Then, run the following command:

    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:** After the environment is created, activate it:

    ```bash
    conda activate SP2025P
    ```

4.  **Run the Notebooks:** Now you can open and run the Jupyter Notebooks (`.ipynb` files) using Jupyter Lab or Jupyter Notebook.

**Environment Details (from `environment.yml`):**

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
    - tensorflow==1.15.2   # original author recommended for tensorflow implementation
    - matplotlib==3.1.2
    - torch==1.13.1
    - torchvision==0.14.1
    - torchaudio==0.13.1
```
**Disclaimer:**

This project was developed for an academic course (CSE 6250) and may not represent production-level code.

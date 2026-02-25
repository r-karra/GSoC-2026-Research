Quark-Gluon Jet Classifier (JAX/Flax)
This repository contains a high-performance implementation of a Quark-Gluon Jet Classifier, developed as a foundational evaluation task for GSoC 2026 under the ML4Sci (Machine Learning for Science) umbrella.

Background
In High Energy Physics (HEP), distinguishing between jets originating from quarks and gluons is a critical task for identifying rare signals at the Large Hadron Collider (LHC).

Quark Jets: Typically have lower particle multiplicity and narrower radiation patterns.

Gluon Jets: Due to a higher color charge, they exhibit higher multiplicity and broader fragmentation.

This project implements a Multilayer Perceptron (MLP) baseline using the Google Research JAX/Flax ecosystem, optimized for XLA (Accelerated Linear Algebra) execution.

Tech Stack
Engine: JAX (Differentiable programming & XLA compilation)

Neural Networks: Flax (Linen API)

Optimization: Optax (Gradient processing)

Dataset: ML4Sci Quarks and Gluons Kaggle Dataset

Key Features
Functional Paradigm: Leverages JAX’s stateless design and explicit PRNG key handling.

XLA Optimization: Uses @jax.jit for Just-In-Time compilation, significantly reducing training latency.

Vectorized Training: Employs jax.vmap for efficient batch processing.

Binary Classification: Optimized for the Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC) metrics—the gold standard in HEP research.

Project Structure

```
├── data/               # Instructions to download the ML4Sci dataset
├── src/
│   ├── model.py        # Flax Linen implementation of the MLP
│   ├── train.py        # JAX-accelerated training loop
│   └── utils.py        # Data normalization and feature engineering
├── notebooks/
│   └── qg_classification_demo.ipynb # Google Colab-ready walkthrough
└── requirements.txt    # JAX, Flax, Optax, and dependencies
```

Performance Benchmark
The model is evaluated based on its ability to separate signal (Quarks) from background (Gluons).

Baseline AUC: ~0.80+ (depending on the specific run-file used).

Target Metric: Maximize the True Positive Rate (TPR) at low False Positive Rates (FPR) for jet tagging applications.

Connection to GSoC 2026
This benchmark serves as a precursor to the Quantum Particle Transformer (Q-ParT) project. By establishing a robust classical baseline in JAX, I am prepared to replace standard linear layers with Variational Quantum Circuits (VQCs) to explore quantum advantage in jet discrimination.

Author: [Rajesh Karra]

B.Sc. Computer Science | Research focusing on Quantum AI & Sequential Data
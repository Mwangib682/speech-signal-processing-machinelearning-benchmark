# Technical Documentation

## Comparative Evaluation of Feature Selection and Machine Learning Algorithms for Multiclass Speech Classification

This directory contains the technical and methodological documentation supporting the implementation and evaluation of the speech signal processing project.

The study is organized into four stages:

### Stage 1 — Audio Preprocessing
Documentation of audio resampling, mono conversion, noise reduction, and amplitude normalization.

### Stage 2 — Acoustic Feature Extraction
Explanation of the extraction of 95 acoustic features, including MFCCs, delta coefficients, RMS energy, Zero-Crossing Rate, spectral characteristics, and fundamental frequency.

### Stage 3 — Feature Selection
Technical comparison of seven feature selection algorithms: ANOVA F-Test, Chi-Square, Mutual Information, mRMR, Recursive Feature Elimination, Genetic Algorithm, and L1 Logistic Regression.

Feature subset sizes of 15, 30, and 45 were evaluated.

### Stage 4 — Machine Learning Classification
Comparative evaluation of eight machine learning classifiers using the selected acoustic features, including SVM, Random Forest, KNN, MLP, Logistic Regression, Decision Tree, Gradient Boosting, and Gaussian Naive Bayes.

## Planned Documentation

- `methodology.md` — Dataset preparation, experimental design, and processing pipeline.
- `feature_selection.md` — Feature selection algorithms, configurations, and evaluation.
- `classification.md` — Classification algorithms, metrics, and experimental results.
- `reproducibility.md` — Software environment, dataset access, execution instructions, and limitations.


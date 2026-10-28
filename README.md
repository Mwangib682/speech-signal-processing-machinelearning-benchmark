# Comparative Evaluation of Feature Selection and Machine Learning Algorithms for Multiclass Speech Classification

A research code and experimental-results portfolio comparing **seven feature-selection methods** and **eight classical machine-learning classifiers** for a balanced **10-class speech classification** task. The original experiment used **3,000 recordings** (300 per class, labels 0–9).

> **Research status:** The source code and selected original experiment reports are published. The repository-ready scripts have not yet been independently rerun end-to-end. Reported scores are from the original experiment, and the existing test partition is **not** a fresh independent confirmatory test.

## Results at a glance

| Measure | Reported result |
|---|---:|
| Original acoustic features | 95 |
| Selected features | 45 |
| Feature reduction | 52.63% |
| Feature-selection winner | L1 Logistic Regression (45 features) |
| Classification winner (training CV) | SVM with RBF kernel |
| SVM training 5-fold CV macro F1 | 98.59% |
| SVM existing-test accuracy | **98.67%** |
| SVM existing-test macro precision | 98.68% |
| SVM existing-test macro recall | 98.67% |
| SVM existing-test macro F1 | **98.67%** |
| Correct test predictions | 592 / 600 |

The SVM was selected by **training cross-validation**, not by choosing the highest test score.

### Classifier comparison

| Classifier | Existing-test accuracy | Existing-test macro F1 |
|---|---:|---:|
| SVM (RBF) | **98.67%** | **98.67%** |
| K-Nearest Neighbors | 98.33% | 98.33% |
| Gradient Boosting | 97.83% | 97.83% |
| Random Forest | 97.67% | 97.66% |
| Logistic Regression | 97.17% | 97.16% |
| Neural Network (MLP) | 95.50% | 95.50% |
| Gaussian Naive Bayes | 90.00% | 89.98% |
| Decision Tree | 79.83% | 79.66% |

Source: [original classification comparison](results/classification/classification_comparison.csv) and [experiment summary](results/classification/classification_summary.txt).

## Experimental pipeline

1. **Audio preprocessing:** mono audio, 16 kHz resampling, nonstationary noise reduction, peak normalization to 0.95, and 16-bit PCM WAV export. [Source](src/01_audio_preprocessing.py)
2. **Acoustic feature extraction:** 95 features covering MFCC, delta and delta-delta MFCC, RMS, zero-crossing rate, spectral centroid/bandwidth/rolloff/flatness/flux, F0, and voiced fraction. [Source](src/02_feature_extraction.py)
3. **Feature selection:** ANOVA F-test, Chi-square, Mutual Information, mRMR, RFE, Genetic Algorithm, and L1 Logistic Regression; feature counts 15, 30, and 45; 3-fold training CV. [Source](src/03_feature_selection.py)
4. **Classification:** SVM, Random Forest, KNN, MLP, Logistic Regression, Decision Tree, Gradient Boosting, and Gaussian Naive Bayes; hyperparameter grid search with 5-fold training CV. [Source](src/04_machine_learning_classification.py)

## Repository contents

```text
src/                         # Four numbered pipeline scripts
data/README.md               # Dataset structure and access notes
results/feature_selection/   # Original feature-selection reports
results/classification/      # Original classifier comparison and reports
figures/README.md            # Figure documentation
docs/README.md               # Research documentation
requirements.txt             # Python package dependencies
```

See [feature-selection results](results/feature_selection/feature_selection_results.csv), [feature importance scores](results/feature_selection/feature_importance_scores.csv), [selected feature names by method](results/feature_selection/selected_features.csv), [classification tuning results](results/classification/hyperparameter_tuning_results.csv), and [per-class classifier report](results/classification/classification_report.csv).

## Running the published pipeline locally

Use Python in a virtual environment. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Organize your **locally obtained** dataset as `data/raw/0/`, `data/raw/1/`, …, `data/raw/9/`, with audio files in the corresponding class directories. Then, from the repository root:

```bash
python src/01_audio_preprocessing.py
python src/02_feature_extraction.py
python src/03_feature_selection.py
python src/04_machine_learning_classification.py
```

Outputs are generated locally under `data/processed/`, `data/voice_features.csv`, `data/feature_selection_results/`, and `data/classification_results/`. These paths are ignored by Git to avoid accidentally publishing raw recordings or large generated files.

**Note:** Stage 3 uses computationally intensive custom mRMR and Genetic Algorithm searches. Runtime and exact results can depend on package versions and hardware. This README describes the published workflow, not a verified fresh rerun.

## Evaluation design, integrity, and limitations

- The original split is stratified **80% training (2,400) / 20% test (600)** with random seed 42.
- Feature selection is fitted using training data, but Stage 4 performs cross-validation **after** selecting features using the full Stage 3 training partition. Consequently, Stage 4 CV estimates are **not nested** and may be optimistic.
- Stage 3 already evaluated its winning selection method on the same test partition later used by Stage 4. Stage 4 test metrics are therefore **descriptive**, not an independent confirmatory estimate.
- No verified speaker IDs are available. Recordings from the same speaker may appear across train and test; speaker-independent generalization **cannot be claimed**.
- Stage 4 reconstructs the split using the same seed and row order. Reordering or filtering the Stage 3 exported dataset before Stage 4 would break split consistency.
- The original audio dataset is not redistributed here. Its source, provenance, and license require verification before any public release. Reuse of any third-party recordings or feature-level data must follow their applicable permissions.
- The reported scores should not be interpreted as deployment performance or generalization to unseen speakers, microphones, or recording environments.

## Future work

Evaluate speaker/group-disjoint splits where speaker metadata exists, perform fully nested feature selection and model tuning, document the dataset license and precise environment versions, and validate the end-to-end repository workflow on a fresh installation. Deep-learning/STFT work is outside the scope of this four-stage benchmark.

## Research outputs

This repository documents an academic comparative experiment and provides its implementation scripts and selected original reports. The reported results have not been independently reproduced from this published repository. If reusing the results, cite the repository and explicitly retain the evaluation limitations above.

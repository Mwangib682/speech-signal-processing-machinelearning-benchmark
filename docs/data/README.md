# Dataset Documentation

## Overview

This project uses a balanced multiclass audio dataset containing 3,000 speech recordings distributed across 10 classes.

The dataset supports the evaluation of acoustic feature extraction, feature selection, and traditional machine learning classification algorithms.

## Dataset Characteristics

| Property | Description |
|---|---|
| Total recordings | 3,000 |
| Number of classes | 10 |
| Class labels | 0–9 |
| Recordings per class | 300 |
| Target sampling rate | 16,000 Hz |
| Processed audio format | WAV |
| Audio channels | Mono |
| Classification task | Multiclass speech classification |

## Dataset Organization

The audio recordings are organized into class-specific directories:

```text
dataset/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
```

Each class directory contains 300 recordings.

## Data Preprocessing

The audio preprocessing pipeline includes:

1. Conversion to mono.
2. Resampling to 16 kHz.
3. Background noise reduction.
4. Peak amplitude normalization.
5. Saving standardized WAV recordings.

## Feature Dataset

After preprocessing, 95 acoustic features were extracted from each recording and organized into a structured feature dataset for subsequent feature selection and classification.

## Dataset Availability

The original audio recordings are not included in this repository at this stage.

Dataset source, access permissions, licensing information, and download instructions will be documented after verification.

## Reproducibility

Researchers wishing to reproduce the experiments should preserve the class labels, verify the number of recordings in each class, and follow the preprocessing and evaluation procedures described in the project documentation.

The train/test partition and feature selection procedures must be applied consistently to prevent data leakage.

# Experimental Results

This directory contains the experimental results obtained from the comparative evaluation of feature selection and machine learning classification algorithms for multiclass speech classification.

## 1. Feature Selection Results

Seven feature selection algorithms were evaluated using feature subset sizes of 15, 30, and 45.

The selected configuration was L1 Logistic Regression with 45 features, achieving approximately 98.46% cross-validation macro F1-score and reducing feature dimensionality by 52.63%.

## 2. Machine Learning Classification Results

Eight classification algorithms were evaluated using 45 selected acoustic features.

| Classifier | Test Accuracy | Macro F1-Score |
|---         |---:           |---:            |
| SVM (RBF)  |        98.67% |         98.67% |
| KNN        |        98.33% |         98.33% |
| Gradient Boosting | 97.83% |         97.83% |
| Random Forest |     97.67% |         97.66% |
| Logistic Regression| 97.17%|         97.16% |
| MLP |               95.50% |         95.50% |
| Gaussian Naive Bayes|90.00%|         89.98% |
| Decision Tree |     79.83% |         79.66% |

## 3. Supporting Files

The original experiment outputs, including feature selection comparison tables, classification reports, confusion matrices, and performance visualizations, will be organized and added to this directory.

## Note

These results represent the reported experimental evaluations. Reproducibility details, including the train/test split and feature-selection procedure, will be documented alongside the source code.

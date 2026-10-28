"""Stage 4: classification of 10 voice classes using 45 Stage-3 features.

Install: python -m pip install numpy pandas scipy scikit-learn matplotlib joblib
Run from repository root: python src/04_machine_learning_classification.py
Outputs: data/classification_results/ (generated locally; not committed).

Methodology:
- Reproduces Stage 3's stratified 80/20 split (seed 42) from its
  selected_feature_dataset.csv, preserving row order.
- Tunes eight models on the 80% training partition using 5-fold CV.
- Picks the winner by training CV macro F1, then evaluates all eight on
  the existing 20% test partition for descriptive comparisons.
- Stage 3 feature selection was performed on the entire 80% training
  partition, so the Stage 4 CV scores are conditional on this previously
  selected feature subset and may be optimistic.
- The test partition was already examined in Stage 3: this is not a new
  independent confirmatory test. Do not select a new winner based on test.
"""
from __future__ import annotations

import json
import time
import traceback
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / 'data' / 'feature_selection_results' / 'selected_feature_dataset.csv'
OUTPUT_FOLDER = PROJECT_ROOT / 'data' / 'classification_results'
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = 1  # safest for Windows/PyCharm; increase to -1 if enough RAM


def models():
    """All scaling and imputation fitted within each training CV fold."""
    def pipe(clf, scale=False):
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', clf))
        return Pipeline(steps)

    return {
        'SVM (RBF)': (
            pipe(SVC(kernel='rbf', cache_size=400), scale=True),
            {'classifier__C': [0.5, 1.0, 5.0],
             'classifier__gamma': ['scale', 0.01]}),
        'Random Forest': (
            pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            {'classifier__n_estimators': [150, 300],
             'classifier__max_depth': [None, 18]}),
        'K-Nearest Neighbors': (
            pipe(KNeighborsClassifier(), scale=True),
            {'classifier__n_neighbors': [3, 5, 9],
             'classifier__weights': ['uniform', 'distance']}),
        'Neural Network (MLP)': (
            pipe(MLPClassifier(random_state=RANDOM_STATE, max_iter=450,
                               early_stopping=True, n_iter_no_change=20), scale=True),
            {'classifier__hidden_layer_sizes': [(64,), (128, 64)],
             'classifier__alpha': [0.0001, 0.01]}),
        'Logistic Regression': (
            pipe(LogisticRegression(max_iter=2500, solver='lbfgs',
                                    random_state=RANDOM_STATE), scale=True),
            {'classifier__C': [0.1, 1.0, 10.0]}),
        'Decision Tree': (
            pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {'classifier__max_depth': [8, 16, None],
             'classifier__min_samples_leaf': [1, 3]}),
        'Gradient Boosting': (
            pipe(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            {'classifier__max_iter': [100, 200],
             'classifier__learning_rate': [0.05, 0.1],
             'classifier__max_leaf_nodes': [15]}),
        'Gaussian Naive Bayes': (
            pipe(GaussianNB(), scale=True),
            {'classifier__var_smoothing': [1e-9, 1e-7, 1e-5]}),
    }


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / filename, dpi=180, bbox_inches='tight')
    plt.close()


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print('=== STAGE 4: MULTICLASS AUDIO CLASSIFICATION ===', flush=True)
    print('scikit-learn:', sklearn.__version__, flush=True)
    print('Reading:', INPUT_CSV, flush=True)
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f'Missing {INPUT_CSV}\nRun Stage 3 first to generate '
            'selected_feature_dataset.csv in data/feature_selection_results/.'
        )
    df = pd.read_csv(INPUT_CSV)
    if 'class_label' not in df.columns:
        raise ValueError('Input CSV must have a class_label column.')
    feature_names = [c for c in df.columns if c != 'class_label']
    X = df[feature_names].apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df['class_label']
    if y.isna().any():
        raise ValueError('class_label contains missing values.')
    if X.shape[1] != 45:
        print(f'WARNING: Found {X.shape[1]} features, expected 45.', flush=True)
    if y.value_counts().min() < CV_FOLDS + 2:
        raise ValueError('Insufficient recordings per class for CV.')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    classes = np.sort(y.unique())
    print(f'Data: {len(df)} recordings, {len(classes)} classes, {len(feature_names)} features', flush=True)
    print(f'Train: {len(train_idx)} | Test: {len(test_idx)} | 5-fold CV', flush=True)

    # The split manifest allows future comparisons to use exactly the same rows.
    pd.DataFrame({
        'original_row_index': np.r_[train_idx, test_idx],
        'partition': ['train'] * len(train_idx) + ['test'] * len(test_idx),
        'class_label': y.iloc[np.r_[train_idx, test_idx]].to_numpy(),
    }).to_csv(OUTPUT_FOLDER / 'data_split_manifest.csv', index=False)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, tuning_rows, report_rows, predictions = [], [], [], []
    matrices = {}
    fitted = {}
    start_all = time.perf_counter()

    for name, (pipeline, params) in models().items():
        print(f'\nTraining {name} ...', flush=True)
        started = time.perf_counter()
        search = GridSearchCV(
            estimator=pipeline, param_grid=params, scoring='f1_macro',
            cv=cv, n_jobs=N_JOBS, refit=True, return_train_score=False,
            error_score='raise', verbose=0,
        )
        # Do not suppress convergence warnings: they signal model tuning issues.
        search.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started
        pred_start = time.perf_counter()
        y_pred = search.predict(X_test)
        prediction_seconds = time.perf_counter() - pred_start
        best_idx = search.best_index_
        cv_mean = float(search.cv_results_['mean_test_score'][best_idx])
        cv_std = float(search.cv_results_['std_test_score'][best_idx])
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        row = {
            'algorithm': name,
            'cv_macro_f1_mean': cv_mean,
            'cv_macro_f1_std': cv_std,
            'test_accuracy': acc,
            'test_macro_precision': macro_precision,
            'test_macro_recall': macro_recall,
            'test_macro_f1': macro_f1,
            'training_and_tuning_seconds': training_seconds,
            'test_prediction_seconds': prediction_seconds,
            'best_parameters': json.dumps(search.best_params_, sort_keys=True),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(OUTPUT_FOLDER / 'classification_comparison.csv', index=False)
        fitted[name] = search.best_estimator_
        matrices[name] = confusion_matrix(y_test, y_pred, labels=classes)
        detail = classification_report(
            y_test, y_pred, labels=classes, output_dict=True, zero_division=0
        )
        for cls in classes:
            c = detail[str(cls)]
            report_rows.append({
                'algorithm': name, 'class_label': cls,
                'precision': c['precision'], 'recall': c['recall'],
                'f1_score': c['f1-score'], 'support': c['support'],
            })
        for row_id, actual, predicted in zip(test_idx, y_test, y_pred):
            predictions.append({
                'algorithm': name, 'original_row_index': int(row_id),
                'actual_label': actual, 'predicted_label': predicted,
                'correct': bool(actual == predicted),
            })
        for i, p in enumerate(search.cv_results_['params']):
            tuning_rows.append({
                'algorithm': name,
                'parameters': json.dumps(p, sort_keys=True),
                'mean_cv_macro_f1': search.cv_results_['mean_test_score'][i],
                'std_cv_macro_f1': search.cv_results_['std_test_score'][i],
                'rank': int(search.cv_results_['rank_test_score'][i]),
            })
        pd.DataFrame(tuning_rows).to_csv(OUTPUT_FOLDER / 'hyperparameter_tuning_results.csv', index=False)
        pd.DataFrame(report_rows).to_csv(OUTPUT_FOLDER / 'classification_report.csv', index=False)
        pd.DataFrame(predictions).to_csv(OUTPUT_FOLDER / 'classification_predictions.csv', index=False)
        print(f'CV macro F1={cv_mean:.4f} ± {cv_std:.4f} | '
              f'test accuracy={acc:.4f} | test macro precision={macro_precision:.4f} | '
              f'test macro recall={macro_recall:.4f} | test macro F1={macro_f1:.4f} | '
              f'time={training_seconds:.1f}s', flush=True)

    result_df = pd.DataFrame(results).sort_values(
        ['cv_macro_f1_mean', 'training_and_tuning_seconds'],
        ascending=[False, True]
    ).reset_index(drop=True)
    result_df.to_csv(OUTPUT_FOLDER / 'classification_comparison.csv', index=False)
    winner = result_df.iloc[0]
    best_name = winner['algorithm']
    best_model = fitted[best_name]
    joblib.dump(best_model, OUTPUT_FOLDER / 'best_classification_model.joblib')
    (OUTPUT_FOLDER / 'selected_feature_names.txt').write_text(
        '\n'.join(feature_names) + '\n', encoding='utf-8'
    )
    joblib.dump({'feature_names': feature_names, 'class_labels': classes.tolist(),
                 'model': best_model, 'random_state': RANDOM_STATE},
                OUTPUT_FOLDER / 'best_classification_bundle.joblib')

    # Performance figure: training CV used for model selection, test descriptive only.
    ordered = result_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    yy = np.arange(len(ordered))
    ax.barh(yy - 0.18, ordered['cv_macro_f1_mean'] * 100,
            height=0.35, label='Training CV macro F1')
    ax.barh(yy + 0.18, ordered['test_macro_f1'] * 100,
            height=0.35, label='Existing test macro F1')
    ax.set_yticks(yy, ordered['algorithm'])
    ax.set_xlim(0, 105)
    ax.set_xlabel('Macro F1 (%)')
    ax.set_title('Classifier comparison: CV vs existing test partition')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.2)
    save_figure('classification_performance.png')

    fig, ax = plt.subplots(figsize=(11, 6))
    ordered_time = result_df.sort_values('training_and_tuning_seconds')
    ax.barh(ordered_time['algorithm'], ordered_time['training_and_tuning_seconds'])
    ax.invert_yaxis()
    ax.set_xlabel('Training and hyperparameter tuning time (seconds)')
    ax.set_title('Computational cost by classifier')
    ax.grid(axis='x', alpha=0.2)
    save_figure('classification_training_times.png')

    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    fig.suptitle('Confusion matrices on the existing 600-recording test partition', fontsize=14)
    fig.savefig(OUTPUT_FOLDER / 'classification_confusion_matrices.png',
                dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrices[best_name], display_labels=classes).plot(
        ax=ax, cmap='Blues', colorbar=False, values_format='d'
    )
    ax.set_title(f'CV-selected best classifier: {best_name}')
    save_figure('best_classifier_confusion_matrix.png')

    duration = time.perf_counter() - start_all
    lines = [
        'STAGE 4: AUDIO CLASSIFICATION — EXPERIMENT SUMMARY',
        '=' * 55,
        f'Dataset: {INPUT_CSV}',
        f'Samples: {len(df)} | Classes: {len(classes)} | Features: {len(feature_names)}',
        f'Training: {len(train_idx)} | Existing test: {len(test_idx)}',
        f'Cross-validation: {CV_FOLDS}-fold stratified, seed {RANDOM_STATE}',
        'Model selection criterion: highest TRAINING CV macro F1',
        f'CV-selected winner: {best_name}',
        f'Winning training CV macro F1: {winner["cv_macro_f1_mean"]:.6f}',
        f'Existing test accuracy: {winner["test_accuracy"]:.6f}',
        f'Existing test macro precision: {winner["test_macro_precision"]:.6f}',
        f'Existing test macro recall: {winner["test_macro_recall"]:.6f}',
        f'Existing test macro F1: {winner["test_macro_f1"]:.6f}',
        f'Best parameters: {winner["best_parameters"]}',
        f'Total experiment time: {duration:.1f} seconds',
        '',
        'METHODOLOGICAL LIMITATIONS',
        '1. Feature selection occurred on the Stage-3 full training partition,',
        '   before Stage-4 CV. Stage-4 CV is not fully nested and can be optimistic.',
        '2. The test partition was already examined in Stage 3; test comparisons',
        '   are descriptive, not a fresh independent confirmatory evaluation.',
        '3. No speaker IDs are provided, so speaker-independent splitting',
        '   cannot be guaranteed. If needed, collect speaker IDs and use groups.',
        '4. The bundle expects the exact listed feature names and feature',
        '   extraction conventions; it does not accept raw audio directly.',
        '',
        'MODEL COMPARISON',
        result_df.to_string(index=False),
    ]
    (OUTPUT_FOLDER / 'classification_summary.txt').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )
    print('\n=== FINAL SUMMARY ===', flush=True)
    print(f'CV-selected winner: {best_name}', flush=True)
    print(f'Training CV macro F1: {winner["cv_macro_f1_mean"]:.4f}', flush=True)
    print(f'Existing test accuracy: {winner["test_accuracy"]:.4f}', flush=True)
    print(f'Existing test macro precision: {winner["test_macro_precision"]:.4f}', flush=True)
    print(f'Existing test macro recall: {winner["test_macro_recall"]:.4f}', flush=True)
    print(f'Existing test macro F1: {winner["test_macro_f1"]:.4f}', flush=True)
    print(f'All outputs saved in: {OUTPUT_FOLDER}', flush=True)
    print(f'Elapsed: {duration:.1f}s', flush=True)


if __name__ == '__main__':
    main()
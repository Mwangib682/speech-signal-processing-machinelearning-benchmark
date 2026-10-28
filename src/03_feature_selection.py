"""Seven-method audio feature selection with leakage-safe evaluation.

Fixes: RFE uses multiclass-compatible LinearSVC; L1 uses l1_ratio=1.0
for scikit-learn 1.8+. Existing reports are overwritten on rerun.


Install: python -m pip install numpy pandas scipy scikit-learn matplotlib
Run from repository root: python src/03_feature_selection.py

A held-out test set is used ONCE for the winning method; selection and tuning
are performed only on training folds. If recordings share speakers, supply
speaker IDs and replace StratifiedShuffleSplit with a group-aware split.

The exported selected_feature_dataset.csv contains all rows. Downstream
classification must reuse this stage's split, rather than create a new one,
to preserve held-out test independence.
"""
from pathlib import Path
from collections import Counter
from functools import lru_cache
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC, LinearSVC
from sklearn.inspection import permutation_importance

# ===== SETTINGS =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / 'data' / 'voice_features.csv'
OUTPUT_FOLDER = PROJECT_ROOT / 'data' / 'feature_selection_results'
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
FEATURE_COUNTS = (15, 30, 45)  # test several subset sizes; adjust as needed
GA_POPULATION = 10
GA_GENERATIONS = 5
GA_CANDIDATE_POOL = 60  # must be >= largest requested k (45)

# ===== CUSTOM FEATURE SELECTORS =====
class MRMRSelector(BaseEstimator, TransformerMixin):
    """Greedy mRMR MID: MI(feature; class) - mean MI(feature; selected).
    Continuous features are discretized into quantile bins for redundancy MI.
    All fitting occurs within each training fold.
    """
    def __init__(self, k=30, bins=8, random_state=42):
        self.k, self.bins, self.random_state = k, bins, random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        k = min(self.k, n_features)
        relevance = mutual_info_classif(X, y, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            discrete = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='quantile').fit_transform(X).astype(int)
        from sklearn.metrics import mutual_info_score
        @lru_cache(None)
        def redundancy(i, j):
            return mutual_info_score(discrete[:, i], discrete[:, j])
        selected = []
        available = set(range(n_features))
        for _ in range(k):
            best = max(available, key=lambda j: (relevance[j] - (np.mean([redundancy(min(j, s), max(j, s)) for s in selected]) if selected else 0.0), -j))
            selected.append(best)
            available.remove(best)
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


class GASelector(BaseEstimator, TransformerMixin):
    """Small genetic search, with internal stratified CV and SVM fitness.
    Uses only the training fold supplied to fit().
    """
    def __init__(self, k=30, pool=40, population=10, generations=5, random_state=42):
        self.k, self.pool, self.population, self.generations, self.random_state = k, pool, population, generations, random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        relevance = mutual_info_classif(X, y, random_state=self.random_state)
        candidates = np.argsort(relevance)[::-1][:min(self.pool, X.shape[1])]
        if self.k > len(candidates):
            raise ValueError(f'GA candidate pool {len(candidates)} is smaller than requested k={self.k}')
        k = self.k
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        cache = {}

        def fitness(indices):
            key = tuple(sorted(indices))
            if key not in cache:
                clf = SVC(kernel='rbf', C=1.0, gamma='scale')
                cache[key] = float(np.mean(cross_val_score(clf, X[:, candidates[list(key)]], y, cv=cv, scoring='f1_macro', n_jobs=1)))
            return cache[key]

        population = [rng.choice(len(candidates), size=k, replace=False) for _ in range(self.population)]
        for _ in range(self.generations):
            population.sort(key=fitness, reverse=True)
            elite = population[:max(2, self.population // 3)]
            next_population = [a.copy() for a in elite]
            while len(next_population) < self.population:
                p1, p2 = rng.choice(len(elite), size=2, replace=True)
                pool = np.unique(np.concatenate([elite[p1], elite[p2]]))
                if len(pool) < k:
                    pool = np.unique(np.concatenate([pool, rng.permutation(len(candidates))]))
                child = rng.choice(pool, size=k, replace=False)
                if rng.random() < 0.4 and k < len(candidates):
                    replace_at = rng.integers(k)
                    alternatives = np.setdiff1d(np.arange(len(candidates)), child)
                    child[replace_at] = rng.choice(alternatives)
                next_population.append(child)
            population = next_population
        winner = max(population, key=fitness)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[candidates[winner]] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


def build_pipeline(method, k):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if method == 'Chi-Square':
        steps += [('scale', MinMaxScaler()), ('select', SelectKBest(chi2, k=k))]
    else:
        steps += [('scale', StandardScaler())]
        if method == 'ANOVA F-Test':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'Mutual Information':
            selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE), k=k)
        elif method == 'mRMR':
            selector = MRMRSelector(k=k, random_state=RANDOM_STATE)
        elif method == 'RFE':
            selector = RFE(estimator=LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=RANDOM_STATE), n_features_to_select=k, step=0.2)
        elif method == 'Genetic Algorithm':
            selector = GASelector(k=k, pool=GA_CANDIDATE_POOL, population=GA_POPULATION, generations=GA_GENERATIONS, random_state=RANDOM_STATE)
        elif method == 'L1 Logistic Regression':
            # select exactly k features by absolute coefficient importance
            selector = SelectFromModel(LogisticRegression(l1_ratio=1.0, solver='saga', C=0.3, max_iter=2500, random_state=RANDOM_STATE), threshold=-np.inf, max_features=k)
        else:
            raise ValueError(method)
        steps.append(('select', selector))
    steps.append(('classifier', SVC(kernel='rbf', C=1.0, gamma='scale')))
    return Pipeline(steps)


def save_bar(series, path, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(series))))
    series.iloc[::-1].plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def generate_feature_importance_report(
    X_train, y_train, feature_columns, selection_rows, effects, permutation_result
):
    """Create one row per original feature, using training-only statistical
    scores and descriptive held-out permutation importance.

    IMPORTANT: The 0-100 consensus score is a descriptive composite, not
    a probability or a validated measure of causal importance.
    """
    from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

    names = list(feature_columns)
    train_array = SimpleImputer(strategy="median").fit_transform(X_train)
    scaled = StandardScaler().fit_transform(train_array)
    nonnegative = MinMaxScaler().fit_transform(train_array)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        anova_scores, anova_pvalues = f_classif(scaled, y_train)
        chi_scores, chi_pvalues = chi2(nonnegative, y_train)
    mi_scores = mutual_info_classif(
        scaled, y_train, random_state=RANDOM_STATE
    )

    freq = Counter(row["feature"] for row in selection_rows)
    effect_lookup = {}
    for row in effects:
        name = row["feature"]
        value = float(row["absolute_standardized_mean_difference"])
        effect_lookup[name] = max(effect_lookup.get(name, 0.0), value)

    pi_mean = dict(zip(names, permutation_result.importances_mean))
    pi_std = dict(zip(names, permutation_result.importances_std))

    report = pd.DataFrame({
        "feature": names,
        "anova_f_score": anova_scores,
        "anova_p_value": anova_pvalues,
        "chi_square_score": chi_scores,
        "chi_square_p_value": chi_pvalues,
        "mutual_information_score": mi_scores,
        "selection_frequency": [freq.get(n, 0) for n in names],
        "selection_frequency_percent": [
            100.0 * freq.get(n, 0) / 7 for n in names
        ],
        "max_class_specific_effect_size_top10": [
            effect_lookup.get(n, np.nan) for n in names
        ],
        "test_permutation_importance_macro_f1_drop": [
            pi_mean[n] for n in names
        ],
        "test_permutation_importance_std": [
            pi_std[n] for n in names
        ],
    })

    # Rank normalization is robust to very large score magnitudes.
    # Use only training-derived measures for consensus; test permutation
    # importance is reported separately and NEVER used for ranking.
    rank_columns = [
        "anova_f_score",
        "chi_square_score",
        "mutual_information_score",
        "selection_frequency",
    ]
    normalized_ranks = []
    for col in rank_columns:
        values = pd.to_numeric(report[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
            normalized = pd.Series(0.0, index=report.index)
        else:
            normalized = (
                values.rank(method="average", pct=True).fillna(0.0) * 100.0
            )
        normalized_ranks.append(normalized)

    report["consensus_importance_score_0_100"] = (
        pd.concat(normalized_ranks, axis=1).mean(axis=1)
    )
    report = report.sort_values(
        ["consensus_importance_score_0_100", "feature"],
        ascending=[False, True]
    ).reset_index(drop=True)
    report.insert(0, "rank", np.arange(1, len(report) + 1))
    report.to_csv(
        OUTPUT_FOLDER / "feature_importance_scores.csv",
        index=False,
        float_format="%.8g",
    )

    top = report.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(
        top["feature"], top["consensus_importance_score_0_100"],
        color="#3569a8"
    )
    ax.set_xlim(0, 105)
    ax.set_xlabel("Consensus importance score (0-100; training-based rank composite)")
    ax.set_title("Top 25 most distinctive voice features")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, top["consensus_importance_score_0_100"]):
        ax.text(
            value + 1, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}", va="center", fontsize=8
        )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_FOLDER / "feature_importance_scores.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    (OUTPUT_FOLDER / "feature_importance_readme.txt").write_text(
        "Feature importance report\\n"
        "=========================\\n"
        "ANOVA, Chi-square, Mutual Information and selection frequency "
        "are calculated from the TRAINING data only.\\n"
        "Chi-square is calculated on min-max-scaled continuous features; "
        "interpret it as a nonnegative feature ranking heuristic, not a "
        "classical contingency-table independence test.\\n"
        "Class-specific effect sizes are the largest one-versus-rest "
        "standardized mean differences among each class's top 10; "
        "missing values do not imply zero effect.\\n"
        "Permutation importance is calculated on the HELD-OUT TEST set "
        "for descriptive interpretation of the final selected classifier. "
        "Do not use test importance to choose or tune features.\\n"
        "Consensus score = mean of percentile ranks (0-100) for ANOVA F, "
        "Chi-square, Mutual Information and seven-method selection frequency. "
        "It is a descriptive research-defined composite, not a probability.\\n"
        "Because all methods can select correlated features, selection "
        "frequency and permutation importance should be interpreted cautiously.\\n",
        encoding="utf-8",
    )
    print(
        "Feature importance files saved: feature_importance_scores.csv "
        "and feature_importance_scores.png", flush=True
    )
    return report


def main():
    start = time.perf_counter()
    print('=== FEATURE SELECTION v4: SEVEN METHODS + FEATURE IMPORTANCE REPORT ===', flush=True)
    print('Running script:', Path(__file__).resolve(), flush=True)
    print('scikit-learn version:', __import__('sklearn').__version__, flush=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f'Feature CSV not found: {INPUT_CSV}')
    df = pd.read_csv(INPUT_CSV)
    if 'class_label' not in df.columns:
        raise ValueError('Missing class_label column in CSV')
    # Exclude metadata; preserve actual numeric feature names
    metadata = {'class_label', 'file_name', 'filename', 'file_path', 'path', 'class', 'source_file'}
    feature_columns = [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_columns:
        raise ValueError('No numeric feature columns found')
    if len(feature_columns) != 95:
        print(f'NOTE: Detected {len(feature_columns)} numeric features, expected 95. Review CSV columns.', flush=True)
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = df['class_label']
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if y.value_counts().min() < 5:
        raise ValueError('Each class needs at least 5 samples for the default split and CV.')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    methods = ['ANOVA F-Test', 'Chi-Square', 'Mutual Information', 'mRMR', 'RFE', 'Genetic Algorithm', 'L1 Logistic Regression']
    ks = [k for k in FEATURE_COUNTS if k <= len(feature_columns)]
    if not ks:
        ks = [len(feature_columns)]
    results = []
    print(f'Loaded {len(X)} recordings, {len(feature_columns)} features, {y.nunique()} classes')
    print(f'Train: {len(train_idx)} | untouched test: {len(test_idx)}')
    print('Class counts:', y.value_counts().sort_index().to_dict(), flush=True)
    for method in methods:
        for k in ks:
            t0 = time.perf_counter()
            try:
                pipeline = build_pipeline(method, k)
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1, error_score='raise')
                result = dict(method=method, requested_features=k, cv_macro_f1_mean=float(scores.mean()), cv_macro_f1_std=float(scores.std()), runtime_seconds=round(time.perf_counter()-t0, 2), error='')
                print(f'{method:25s} k={k:2d} | CV macro F1={scores.mean():.4f} ± {scores.std():.4f} | {result["runtime_seconds"]:.1f}s', flush=True)
            except Exception as exc:
                result = dict(method=method, requested_features=k, cv_macro_f1_mean=np.nan, cv_macro_f1_std=np.nan, runtime_seconds=round(time.perf_counter()-t0, 2), error=str(exc))
                print(f'FAILED {method}, k={k}: {exc}', flush=True)
            results.append(result)
            pd.DataFrame(results).to_csv(OUTPUT_FOLDER / 'feature_selection_results.csv', index=False)
    failed = [r for r in results if r['error']]
    if failed:
        raise RuntimeError(f'{len(failed)} selection evaluations failed. Review feature_selection_results.csv; final report withheld.')
    ranking = pd.DataFrame(results).dropna(subset=['cv_macro_f1_mean']).sort_values(['cv_macro_f1_mean', 'requested_features'], ascending=[False, True])
    if ranking.empty:
        raise RuntimeError('All feature selection runs failed. Inspect feature_selection_results.csv.')
    best = ranking.iloc[0]
    best_method, best_k = str(best.method), int(best.requested_features)
    print(f'\nWinner by training CV: {best_method} with {best_k} features')

    # Refit each method at its best k on all training data to report selected names.
    selection_rows = []
    for method in methods:
        subset = ranking[ranking.method == method]
        if subset.empty:
            continue
        chosen_k = int(subset.iloc[0].requested_features)
        fitted = build_pipeline(method, chosen_k).fit(X_train, y_train)
        names = [n for n, use in zip(feature_columns, fitted.named_steps['select'].get_support()) if use]
        for name in names:
            selection_rows.append({'method': method, 'feature': name, 'best_k': chosen_k})
    pd.DataFrame(selection_rows).to_csv(OUTPUT_FOLDER / 'selected_features.csv', index=False)
    frequency = pd.Series(Counter(row['feature'] for row in selection_rows), name='selection_frequency').sort_values(ascending=False)
    frequency.rename_axis('feature').reset_index().to_csv(OUTPUT_FOLDER / 'feature_distinctiveness_ranking.csv', index=False)
    save_bar(frequency.head(20), OUTPUT_FOLDER / 'feature_selection_frequency.png', 'Most frequently selected features', 'Number of algorithms')

    # Class-specific effect sizes: one-vs-rest standardized mean differences.
    imputed_train = SimpleImputer(strategy='median').fit_transform(X_train)
    effects = []
    for cls in sorted(y_train.unique()):
        in_class = imputed_train[y_train.to_numpy() == cls]
        other = imputed_train[y_train.to_numpy() != cls]
        pooled = np.sqrt((in_class.var(axis=0) + other.var(axis=0)) / 2)
        standardized = np.abs(in_class.mean(axis=0) - other.mean(axis=0)) / np.maximum(pooled, 1e-12)
        for j in np.argsort(standardized)[::-1][:10]:
            effects.append({'class_label': cls, 'feature': feature_columns[j], 'absolute_standardized_mean_difference': float(standardized[j])})
    pd.DataFrame(effects).to_csv(OUTPUT_FOLDER / 'class_specific_features.csv', index=False)

    # Only NOW use held-out test set, once, for selected winner.
    final_model = build_pipeline(best_method, best_k).fit(X_train, y_train)
    pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average='macro')
    report = classification_report(y_test, pred, zero_division=0)
    (OUTPUT_FOLDER / 'final_test_report.txt').write_text(f'Method: {best_method}\nSelected features: {best_k}\nAccuracy: {accuracy:.5f}\nMacro F1: {macro_f1:.5f}\n\n{report}', encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'final_test_confusion_matrix.png', dpi=160)
    plt.close()
    selected_names = [n for n, use in zip(feature_columns, final_model.named_steps['select'].get_support()) if use]
    X.loc[:, selected_names].assign(class_label=y).to_csv(OUTPUT_FOLDER / 'selected_feature_dataset.csv', index=False)
    # Permutation importance is assessed on the held-out test set for interpretation only,
    # never used to reselect features or tune the model.
    pi = permutation_importance(final_model, X_test, y_test, n_repeats=3, random_state=RANDOM_STATE, scoring='f1_macro', n_jobs=1)
    pd.DataFrame({'feature': feature_columns, 'test_permutation_importance': pi.importances_mean}).sort_values('test_permutation_importance', ascending=False).to_csv(OUTPUT_FOLDER / 'final_test_permutation_importance.csv', index=False)
    generate_feature_importance_report(
        X_train=X_train,
        y_train=y_train,
        feature_columns=feature_columns,
        selection_rows=selection_rows,
        effects=effects,
        permutation_result=pi,
    )
    print('\n=== FINAL REPORT ===')
    print(f'Recordings: {len(X)} | Classes: {y.nunique()} | Original features: {len(feature_columns)}')
    print(f'Winning method: {best_method} | Selected: {len(selected_names)}')
    print(f'Training CV macro F1: {best.cv_macro_f1_mean:.4f}')
    print(f'Held-out test accuracy: {accuracy:.4f} | macro F1: {macro_f1:.4f}')
    print('Selected feature names:', ', '.join(selected_names))
    print(f'Runtime: {time.perf_counter()-start:.1f} seconds')
    print(f'Reports saved to: {OUTPUT_FOLDER}')


if __name__ == '__main__':
    main()
"""Stand-in experiment: class-balanced calibrated LightGBM vs logistic regression
for rare-positive detection on UCI Adult (income >50K as the rare positive class)."""
import json, os, sys, time
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

SEED = 42
SEEDS = [42, 43, 44]
MAX_ROWS = 200_000
THRESH = 0.5
URL = "https://archive.ics.uci.edu/dataset/2/adult"
OUT = os.path.dirname(os.path.abspath(__file__))


def fetch_with_retry(fn, tries=5):
    """Call a download function, retrying with exponential backoff on HTTP 429/5xx."""
    for i in range(tries):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429 or code >= 500:
                wait = 2 ** i
                print(f"HTTP {code}, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("download failed after retries")


def load_data():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_with_retry(lambda: fetch_ucirepo(id=2))  # ucimlrepo uses requests internally
    X = ds.data.features.copy()
    y = (ds.data.targets.iloc[:, 0].astype(str).str.contains(">50K")).astype(int)
    return X, y


def evaluate(name, model, X_te, y_te):
    """Shared evaluation: F1 on the rare class plus precision/recall at a fixed threshold."""
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= THRESH).astype(int)
    return {
        "f1": f1_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
    }


def preprocessor(num_cols, cat_cols):
    """Shared preprocessing: scaled numerics, one-hot categoricals."""
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def main():
    t0 = time.time()
    print("Loading UCI Adult...")
    X, y = load_data()
    if len(X) > MAX_ROWS:
        idx = X.sample(MAX_ROWS, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
    X = X.fillna("missing")
    n_rows = len(X)
    print(f"Rows: {n_rows}, positive rate: {y.mean():.3f}")
    if n_rows < 1000 or y.nunique() < 2:
        sys.exit("Not enough data for meaningful metrics")

    # Single stratified split; test set touched only at the end.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = preprocessor(num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # Baseline: standard logistic regression, default settings (deterministic given lbfgs).
    print("Fitting baseline (logistic regression)...")
    base = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    base.fit(X_tr, y_tr)
    base_scores = [evaluate("baseline", base, X_te, y_te)]  # deterministic: one run

    # Improved: class-balanced LightGBM, CV hyperparameter search on train only,
    # then sigmoid calibration. Randomized over 3 seeds.
    print("Tuning improved model (LightGBM, CV on train only)...")
    param_dist = {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.03, 0.1],
        "clf__num_leaves": [15, 31, 63],
        "clf__min_child_samples": [20, 50],
    }
    imp_scores, best_params = [], None
    for s in SEEDS:
        lgb = Pipeline([("pre", pre), ("clf", LGBMClassifier(
            class_weight="balanced", random_state=s, n_jobs=2, verbose=-1))])
        search = RandomizedSearchCV(lgb, param_dist, n_iter=6, scoring="f1", cv=cv,
                                    random_state=s, n_jobs=1)
        search.fit(X_tr, y_tr)
        print(f"  seed {s}: best CV F1 {search.best_score_:.4f}")
        cal = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=cv)
        cal.fit(X_tr, y_tr)
        imp_scores.append(evaluate("improved", cal, X_te, y_te))
        best_params = search.best_params_

    def agg(runs, key):
        v = np.array([r[key] for r in runs])
        return float(v.mean()), float(v.std(ddof=0))

    b_f1, _ = agg(base_scores, "f1")
    i_f1, i_std = agg(imp_scores, "f1")
    b_p, _ = agg(base_scores, "precision")
    i_p, _ = agg(imp_scores, "precision")
    b_r, _ = agg(base_scores, "recall")
    i_r, _ = agg(imp_scores, "recall")
    b_std = 0.0  # baseline is deterministic

    metrics = {
        "baseline_score": b_f1, "improved_score": i_f1, "improvement": i_f1 - b_f1,
        "baseline_score_std": b_std, "improved_score_std": i_std,
        "baseline_f1": b_f1, "improved_f1": i_f1,
        "baseline_precision": b_p, "improved_precision": i_p,
        "baseline_recall": b_r, "improved_recall": i_r,
    }
    results = {
        "question": "Can a class-balanced, calibrated gradient-boosting classifier with per-alert feature attributions flag rare positive signals more reliably than a standard logistic-regression alert classifier on a public imbalanced tabular dataset?",
        "method": "Stand-in on UCI Adult (income >50K as rare positive). Single stratified 80/20 split (seed 42). Baseline: default logistic regression on scaled/one-hot features. Improved: class-balanced LightGBM with 3-fold CV randomized search on train, sigmoid calibration; run with seeds 42-44 and averaged. Test set used once per model.",
        "dataset": {
            "name": "UCI Adult income", "source_url": URL, "n_rows": int(n_rows),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "F1 on positive class (income >50K) at threshold 0.5, test set",
        "higher_is_better": True,
        "baseline": "Standard logistic regression (sklearn defaults, scaled numerics + one-hot categoricals)",
        "improved": "LightGBM with balanced class weights, CV hyperparameter search on train, sigmoid-calibrated probabilities",
        "metrics": {k: float(v) for k, v in metrics.items()},
        "notes": (f"Proxy experiment: Adult income is not social-media outbreak data and the positive class is ~25%, "
                  f"not truly rare. Only 3 seeds for the improved model; baseline is deterministic. "
                  f"Per-alert feature attributions were not implemented. Last best LightGBM params: {best_params}. "
                  f"Runtime {time.time() - t0:.0f}s."),
    }
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline (LR)", "improved (LGBM)"], [b_f1, i_f1], yerr=[0, i_std], color=["gray", "steelblue"])
    ax.set_ylabel("F1 (positive class)")
    ax.set_title("Test F1: baseline vs improved")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "f1_comparison.png"), dpi=120)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

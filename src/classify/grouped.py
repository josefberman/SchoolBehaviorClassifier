"""Leave-one-video-out evaluation with class-weighted XGBoost / logistic regression."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.labels import is_transition, ordered_labels


def _inv_weights(y: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv].astype(np.float64)
    return w * (len(y) / w.sum())


def _xgb(n_classes: int) -> XGBClassifier:
    kw = dict(
        eval_metric="mlogloss",
        tree_method="hist",
        max_depth=4,
        n_estimators=200,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    if n_classes > 2:
        kw["objective"] = "multi:softprob"
        kw["num_class"] = n_classes
    else:
        kw["objective"] = "binary:logistic"
    return XGBClassifier(**kw)


def _logreg() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def _mlp() -> Pipeline:
    from sklearn.neural_network import MLPClassifier

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-3,
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )


def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=False,
    )


def _subset_metrics(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict:
    if len(y) == 0:
        return {}
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": {
            lab: float(f1_score(y, pred, average=None, labels=[lab], zero_division=0)[0])
            for lab in labels
        },
        "label_counts": {lab: int(np.sum(y == lab)) for lab in labels},
    }


def grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model: str = "xgb",
    feature_names: list[str] | None = None,
    permute_repeats: int = 0,
) -> dict:
    """Leave-one-group-out. groups are video ids (strings)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups)
    # Replace non-finite with column medians (train-safe: done inside the fold).
    labels = ordered_labels(set(y))
    group_ids = [g for g in sorted(set(groups.tolist()), key=str)]
    oof_true: list[str] = []
    oof_pred: list[str] = []
    oof_group: list[str] = []
    fold_rows = []
    importances = defaultdict(list)

    for held in group_ids:
        tr = groups != held
        te = groups == held
        if te.sum() == 0 or tr.sum() == 0:
            continue
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for mat in (Xtr, Xte):
            inds = ~np.isfinite(mat)
            if inds.any():
                mat[inds] = np.take(med, np.where(inds)[1])

        present = ordered_labels(set(ytr))
        if len(present) < 2:
            continue
        le = LabelEncoder()
        le.fit(present)
        # Drop test labels the train fold never saw — still count them as errors
        # by mapping to a dummy prediction after inverse_transform of in-vocab rows.
        known = np.array([yi in le.classes_ for yi in yte])
        ytr_enc = le.transform(ytr)
        n_cls = len(le.classes_)
        if model == "logreg":
            clf = _logreg()
            clf.fit(Xtr, ytr_enc, clf__sample_weight=_inv_weights(ytr_enc))
            pred_enc = clf.predict(Xte)
        elif model == "mlp":
            clf = _mlp()
            clf.fit(Xtr, ytr_enc)
            pred_enc = clf.predict(Xte)
        elif model == "hgb":
            clf = _hgb()
            clf.fit(Xtr, ytr_enc, sample_weight=_inv_weights(ytr_enc))
            pred_enc = clf.predict(Xte)
        else:
            clf = _xgb(n_cls)
            clf.fit(Xtr, ytr_enc, sample_weight=_inv_weights(ytr_enc))
            pred_enc = clf.predict(Xte)
        pred = np.array([""] * len(yte), dtype=object)
        pred[known] = le.inverse_transform(np.clip(pred_enc[known], 0, n_cls - 1))
        # Unknown-class test rows: force a wrong label so they hurt recall.
        pred[~known] = "__unseen__"

        oof_true.extend(yte.tolist())
        oof_pred.extend(pred.tolist())
        oof_group.extend([str(held)] * int(te.sum()))
        fold_rows.append(
            {
                "held_out": str(held),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                **_subset_metrics(yte, pred, labels),
            }
        )

        if permute_repeats and feature_names and model == "xgb" and known.any():
            base = float(
                f1_score(yte[known], pred[known], average="macro", labels=present, zero_division=0)
            )
            rng = np.random.default_rng(42)
            for j, fname in enumerate(feature_names):
                drops = []
                for _ in range(permute_repeats):
                    Xp = Xte.copy()
                    rng.shuffle(Xp[:, j])
                    pe = clf.predict(Xp)
                    pp = np.array([""] * len(yte), dtype=object)
                    pp[known] = le.inverse_transform(np.clip(pe[known], 0, n_cls - 1))
                    pp[~known] = "__unseen__"
                    drops.append(
                        base
                        - float(
                            f1_score(
                                yte[known],
                                pp[known],
                                average="macro",
                                labels=present,
                                zero_division=0,
                            )
                        )
                    )
                importances[fname].append(float(np.mean(drops)))

    y_t = np.array(oof_true, dtype=object)
    y_p = np.array(oof_pred, dtype=object)
    report = {
        "model": model,
        "n": int(len(y_t)),
        "n_groups": len(group_ids),
        "groups": group_ids,
        "labels": labels,
        "folds": fold_rows,
        "confusion_matrix": confusion_matrix(y_t, y_p, labels=labels).tolist() if len(y_t) else [],
        "classification_report": (
            classification_report(y_t, y_p, labels=labels, zero_division=0, output_dict=True)
            if len(y_t)
            else {}
        ),
    }
    report.update(_subset_metrics(y_t, y_p, labels))
    base_m = np.array([not is_transition(yi) for yi in y_t])
    trans_m = np.array([is_transition(yi) for yi in y_t])
    if base_m.any():
        bl = ordered_labels({yi for yi in y_t[base_m]})
        report["stable"] = _subset_metrics(y_t[base_m], y_p[base_m], bl)
    if trans_m.any():
        tl = ordered_labels({yi for yi in y_t[trans_m]})
        report["transient"] = _subset_metrics(y_t[trans_m], y_p[trans_m], tl)
    if importances:
        report["permutation_importance"] = {
            k: float(np.mean(v)) for k, v in sorted(importances.items(), key=lambda kv: -np.mean(kv[1]))
        }
    return report


def collapse_transients(y: np.ndarray) -> np.ndarray:
    out = np.empty(len(y), dtype=object)
    for i, yi in enumerate(y):
        out[i] = "transient" if is_transition(str(yi)) else str(yi)
    return out


def binary_transient(y: np.ndarray) -> np.ndarray:
    return np.array(["transient" if is_transition(str(yi)) else "stable" for yi in y], dtype=object)

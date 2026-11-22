#!/usr/bin/env python3
"""Figures for the research report (ablation bars, confusion, importance)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "research" / "results"
FIG = ROOT / "research" / "figures"


def _load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def ablation_bars(summary: dict, task: str, path: Path) -> None:
    block = summary["tasks"][task]
    names, f1s, trans = [], [], []
    for k, v in block.items():
        if not isinstance(v, dict) or "macro_f1" not in v:
            continue
        names.append(k)
        f1s.append(v["macro_f1"])
        trans.append(v.get("transient", {}).get("macro_f1", np.nan))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(names))
    ax.bar(x - 0.18, f1s, 0.36, label="macro-F1", color="#1f77b4")
    if np.isfinite(trans).any():
        ax.bar(x + 0.18, trans, 0.36, label="transient macro-F1", color="#d35400")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_title(f"Grouped-CV ablations  ({task}, w={summary.get('window')})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion(summary: dict, task: str, abl: str, path: Path) -> None:
    block = summary["tasks"][task][abl]
    labels = block.get("labels") or []
    cm = np.asarray(block.get("confusion_matrix") or [], dtype=float)
    if cm.size == 0:
        return
    row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 3.5, 0.55 * len(labels) + 3.0))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    f1 = block.get("macro_f1", float("nan"))
    ax.set_title(f"{task} / {abl}  macro-F1={f1:.3f}")
    for i in range(row.shape[0]):
        for j in range(row.shape[1]):
            ax.text(j, i, f"{100 * row[i, j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance(summary: dict, task: str, abl: str, path: Path, top: int = 20) -> None:
    imp = summary["tasks"][task][abl].get("permutation_importance") or {}
    if not imp:
        return
    items = list(imp.items())[:top]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(names) + 1.5))
    ax.barh(names, vals, color="#2c3e50")
    ax.set_xlabel("Δ macro-F1 when permuted")
    ax.set_title(f"Permutation importance  {task}/{abl}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fname in sorted(RES.glob("feature_ablation_w*.json")):
        summary = json.loads(fname.read_text())
        tag = fname.stem.replace("feature_ablation_", "")
        for task in summary.get("tasks", {}):
            ablation_bars(summary, task, FIG / f"ablation_{task}_{tag}.png")
            for abl in ("baseline", "all", "physical", "nonphysical", "baseline+spectral"):
                if abl in summary["tasks"][task]:
                    confusion(summary, task, abl, FIG / f"cm_{task}_{abl}_{tag}.png")
                    importance(summary, task, abl, FIG / f"imp_{task}_{abl}_{tag}.png")
    inf = RES / "inferred_config_cv.json"
    if inf.exists():
        data = json.loads(inf.read_text())
        wrap = {
            "window": "annotated_config",
            "tasks": {"stable_inferred": {"inferred_weights": data.get("xgb", data)}},
        }
        if "macro_f1" in wrap["tasks"]["stable_inferred"]["inferred_weights"]:
            confusion(wrap, "stable_inferred", "inferred_weights", FIG / "cm_inferred_weights.png")
            importance(wrap, "stable_inferred", "inferred_weights", FIG / "imp_inferred_weights.png")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()

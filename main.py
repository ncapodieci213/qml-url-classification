"""
Classical vs. quantum machine learning for phishing-URL detection.

Dataset: PhiUSIIL Phishing URL dataset (UCI ML Repository, id=967)
        label 1 = legitimate, label 0 = phishing
Models: Logistic Regression, Random Forest and an RBF SVM  vs.  a Variational
        Quantum Classifier (VQC) run on Qiskit's Aer simulator.

Usage: python main.py            # writes everything to ./results/

Precision / recall / F1 treat PHISHING as the positive class.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to disk
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo

from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2

N_QUBITS = 10             # one qubit per PCA component
QUANTUM_TRAIN_SIZE = 800  # samples seen by the VQC and "matched" baselines
QUANTUM_TEST_SIZE = 1000  # shared test subsample for comparison
VQC_MAXITER = 100         # COBYLA spends ~(n_weights + 1) evaluations initialising
VQC_SHOTS = 1024
FEATURE_MAP_REPS = 1
ANSATZ_REPS = 1

PHISHING, LEGITIMATE = 0, 1
CLASS_NAMES = {PHISHING: "Phishing", LEGITIMATE: "Legitimate"}
CLASS_COLORS = {PHISHING: "#d62728", LEGITIMATE: "#1f77b4"}

VQC_NAME = "VQC (quantum) [matched]"

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass
class Data:
    # classical train/test split (standardised)
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    urls_test: np.ndarray | None


@dataclass
class QuantumData:
    # PCA-reduced, [0, 1]-scaled, subsampled data shared by VQC and matched baselines
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_idx: np.ndarray  # positions of the test subsample inside the full test set


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def defang(url: object) -> str:
    # in format of hxxps://example[.]com; exported phishing URLs are inert if pasted or auto-linked
    return str(url).replace("http", "hxxp", 1).replace(".", "[.]")


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{filename}")


def clean_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def model_colors(names: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(names)}


def fmt_seconds(seconds: float) -> str:
    if seconds < 0.01:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s" if seconds < 60 else f"{seconds / 60:.1f} min"


def stratified_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    # indices of a class-balanced-by-proportion subsample of size n (sorted)
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def bootstrap_accuracy_ci(correct: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    # 95% bootstrap CI for accuracy
    rng = np.random.default_rng(RANDOM_SEED)
    n, p = len(correct), float(np.mean(correct))
    boot = rng.binomial(n, p, size=n_boot) / n
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


# Data loading and preprocessing
def load_data() -> Data:
    print("Loading PhiUSIIL dataset from the UCI ML Repository ...")
    try:
        dataset = fetch_ucirepo(id=967)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection and that "
            "`ucimlrepo` is installed (pip install ucimlrepo)."
        ) from exc

    X = dataset.data.features
    y = dataset.data.targets.squeeze()
    print(f"  {X.shape[0]:,} rows, {X.shape[1]} columns")

    # keep numeric features only
    X_num = X.select_dtypes(include=np.number)
    skipped = [c for c in X.columns if c not in X_num.columns]
    print(f"  {X_num.shape[1]} numeric features kept; skipped text columns: {skipped}")

    keep = (X_num.notna().all(axis=1) & y.notna()).to_numpy()
    print(f"  rows dropped for missing values: {int((~keep).sum())}")
    X_num, y = X_num[keep], y[keep].astype(int)
    urls = X.loc[keep, "URL"].to_numpy() if "URL" in X.columns else None

    print("  class balance:")
    for label, count in y.value_counts().sort_index().items():
        print(f"    {CLASS_NAMES[int(label)]:<11}: {count:>9,}  ({count / len(y):.1%})")

    y_arr, X_arr = y.to_numpy(), X_num.to_numpy(dtype=float)
    idx_train, idx_test = train_test_split(
        np.arange(len(y_arr)), test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_arr
    )
    scaler = StandardScaler().fit(X_arr[idx_train])  # fit on train only -> no leakage
    print(f"  train: {len(idx_train):,} samples | test: {len(idx_test):,} samples")

    return Data(
        X_train=scaler.transform(X_arr[idx_train]),
        X_test=scaler.transform(X_arr[idx_test]),
        y_train=y_arr[idx_train],
        y_test=y_arr[idx_test],
        feature_names=X_num.columns.tolist(),
        urls_test=urls[idx_test] if urls is not None else None,
    )


def prepare_quantum_inputs(data: Data) -> tuple[QuantumData, PCA]:
    # PCA -> [0, 1] scaling -> stratified subsample, for the VQC and matched baselines
    n_scree = min(30, data.X_train.shape[1])
    if N_QUBITS > n_scree:
        raise ValueError(f"N_QUBITS={N_QUBITS} exceeds the {n_scree} available components")

    pca = PCA(n_components=n_scree, random_state=RANDOM_SEED).fit(data.X_train)
    Z_train = pca.transform(data.X_train)[:, :N_QUBITS]
    Z_test = pca.transform(data.X_test)[:, :N_QUBITS]
    explained = pca.explained_variance_ratio_[:N_QUBITS].sum()
    print(f"\nPCA: {data.X_train.shape[1]} features -> {N_QUBITS} components "
          f"({explained:.1%} of variance kept)")

    lo, hi = np.percentile(Z_train, [1, 99], axis=0)  # fit on train only
    span = np.maximum(hi - lo, 1e-12)
    Z_train = np.clip((Z_train - lo) / span, 0, 1)
    Z_test = np.clip((Z_test - lo) / span, 0, 1)

    tr = stratified_indices(data.y_train, QUANTUM_TRAIN_SIZE, RANDOM_SEED)
    te = stratified_indices(data.y_test, QUANTUM_TEST_SIZE, RANDOM_SEED)
    print(f"Quantum subsample: {len(tr):,} train / {len(te):,} test samples")

    qd = QuantumData(Z_train[tr], data.y_train[tr], Z_test[te], data.y_test[te], te)
    return qd, pca


# Models
def predict_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def score_model(y_train, pred_train, y_test, pred_test, scores, fit_seconds, scope) -> dict:
    correct = pred_test == y_test
    ci_low, ci_high = bootstrap_accuracy_ci(correct)
    return {
        "scope": scope,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_accuracy": accuracy_score(y_train, pred_train),
        "test_accuracy": float(correct.mean()),
        "acc_ci_low": ci_low,
        "acc_ci_high": ci_high,
        "phish_precision": precision_score(y_test, pred_test, pos_label=PHISHING, zero_division=0),
        "phish_recall": recall_score(y_test, pred_test, pos_label=PHISHING, zero_division=0),
        "phish_f1": f1_score(y_test, pred_test, pos_label=PHISHING, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "train_seconds": fit_seconds,
        "y_test": y_test,
        "y_pred": pred_test,
        "scores": scores,
    }


def train_sklearn_models(models: dict, X_train, y_train, X_test, y_test, scope: str) -> dict:
    results = {}
    for name, model in models.items():
        key = f"{name} [{scope}]"
        print(f"  training {key} ...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        res = score_model(
            y_train, model.predict(X_train), y_test, model.predict(X_test),
            predict_scores(model, X_test), fit_seconds, scope,
        )
        res["model"] = model
        results[key] = res
        print(f"    test acc {res['test_accuracy']:.4f} | phishing recall "
              f"{res['phish_recall']:.4f} | {fmt_seconds(fit_seconds)}")
    return results


def classical_models(include_svm: bool) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    }
    if include_svm:  # kernel SVM is O(n^2); only feasible on the small matched subsample
        models["SVM (RBF)"] = SVC(kernel="rbf", random_state=RANDOM_SEED)
    return models


def train_vqc(qd: QuantumData) -> dict:
    print(f"  training {VQC_NAME}: {N_QUBITS} qubits, {len(qd.y_train):,} samples, "
          f"COBYLA maxiter={VQC_MAXITER}, {VQC_SHOTS} shots ...")
    algorithm_globals.random_seed = RANDOM_SEED
    loss_history: list[float] = []

    backend = AerSimulator()
    vqc = VQC(
        feature_map=zz_feature_map(N_QUBITS, reps=FEATURE_MAP_REPS),
        ansatz=real_amplitudes(N_QUBITS, reps=ANSATZ_REPS),
        optimizer=COBYLA(maxiter=VQC_MAXITER),
        sampler=AerSampler(default_shots=VQC_SHOTS, seed=RANDOM_SEED),
        pass_manager=generate_preset_pass_manager(optimization_level=1, backend=backend),
        callback=lambda weights, loss: loss_history.append(float(loss)),
    )

    t0 = time.perf_counter()
    vqc.fit(qd.X_train, qd.y_train)
    fit_seconds = time.perf_counter() - t0

    res = score_model(
        qd.y_train, vqc.predict(qd.X_train), qd.y_test, vqc.predict(qd.X_test),
        vqc.predict_proba(qd.X_test)[:, 1], fit_seconds, "matched",
    )
    res["loss_history"] = loss_history
    print(f"    test acc {res['test_accuracy']:.4f} | phishing recall "
          f"{res['phish_recall']:.4f} | {fmt_seconds(fit_seconds)}")
    return {VQC_NAME: res}


# Figures
def plot_class_distribution(data: Data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (split, y) in zip(axes, [("Train", data.y_train), ("Test", data.y_test)]):
        labels, counts = np.unique(y, return_counts=True)
        bars = ax.bar(
            [CLASS_NAMES[int(l)] for l in labels], counts,
            color=[CLASS_COLORS[int(l)] for l in labels], width=0.55,
        )
        ax.bar_label(bars, labels=[f"{c:,} ({c / len(y):.1%})" for c in counts], padding=3)
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_title(f"{split} split ({len(y):,} samples)", fontweight="bold")
        ax.set_ylabel("URLs")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        clean_axes(ax)
    fig.suptitle("Class distribution: phishing vs. legitimate URLs", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "01_class_distribution.png")


def plot_feature_correlation(data: Data, top_n: int = 15) -> None:
    # features most correlated with the label, and how correlated they are with each other
    df = pd.DataFrame(data.X_train, columns=data.feature_names)
    top = df.corrwith(pd.Series(data.y_train)).abs().dropna().nlargest(top_n).index.tolist()
    df["label"] = data.y_train
    corr = df[top + ["label"]].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), cmap="coolwarm", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.3,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(f"Top {top_n} features by |correlation| with the label (label 1 = legitimate)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "02_feature_correlation.png")


def plot_pca_scree(pca: PCA) -> None:
    ratios = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratios)
    comps = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(comps, ratios, color="#1f77b4", alpha=0.75, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(comps, cumulative, color="#d62728", marker="o", ms=4, lw=2, label="Cumulative")
    ax2.axvline(N_QUBITS + 0.5, color="gray", ls="--", lw=1.2,
                label=f"Kept for the circuit ({N_QUBITS} qubits)")
    ax2.set_ylim(0, 105)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Individual explained variance (%)")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA scree plot", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    fig.tight_layout()
    save_figure(fig, "03_pca_scree.png")


def plot_confusion_matrices(results: dict) -> None:
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    labels = [CLASS_NAMES[PHISHING], CLASS_NAMES[LEGITIMATE]]
    for ax, (name, res) in zip(axes.flat, results.items()):
        cm = confusion_matrix(res["y_test"], res["y_pred"], labels=[PHISHING, LEGITIMATE])
        sns.heatmap(  # colour = row-normalised rate, annotation = raw count
            cm / cm.sum(axis=1, keepdims=True), annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
            cbar=False, xticklabels=labels, yticklabels=labels, linewidths=0.5,
            linecolor="white", annot_kws={"size": 12}, ax=ax,
        )
        ax.set_title(f"{name}\n(n = {len(res['y_test']):,})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion matrices (colour = share of each true class)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "04_confusion_matrices.png")


def plot_metrics_comparison(results: dict) -> None:
    metrics = [
        ("test_accuracy", "Accuracy"),
        ("phish_precision", "Precision\n(phishing)"),
        ("phish_recall", "Recall\n(phishing)"),
        ("phish_f1", "F1\n(phishing)"),
        ("roc_auc", "ROC AUC"),
    ]
    names = list(results)
    colors = model_colors(names)
    x, width = np.arange(len(metrics)), 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, name in enumerate(names):
        bars = ax.bar(x + i * width, [results[name][k] for k, _ in metrics], width,
                      label=name, color=colors[name], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([label for _, label in metrics], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.set_title("Test-set performance (phishing = positive class)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "05_metrics_comparison.png")


def plot_roc_curves(results: dict) -> None:
    colors = model_colors(list(results))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.50)")
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["scores"])
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {res['roc_auc']:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(ls="--", alpha=0.4)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "06_roc_curves.png")


def plot_training_time(results: dict) -> None:
    names = list(results)
    times = [results[n]["train_seconds"] for n in names]
    colors = model_colors(names)

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(names) + 2))
    bars = ax.barh(names, times, color=[colors[n] for n in names], log=True)
    ax.bar_label(bars, labels=[fmt_seconds(t) for t in times], padding=4)
    ax.set_xlim(left=min(times) / 3, right=max(times) * 5)
    ax.invert_yaxis()
    ax.set_xlabel("Training time in seconds (log scale)")
    ax.set_title("Training time", fontweight="bold")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "07_training_time.png")


def plot_vqc_convergence(results: dict) -> None:
    history = np.asarray(results[VQC_NAME].get("loss_history", []))
    if history.size == 0:
        print("  no VQC loss history recorded; skipping convergence plot")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history, color="#d62728", alpha=0.35, lw=1, label="Objective at each evaluation")
    ax.plot(np.minimum.accumulate(history), color="#d62728", lw=2, label="Best so far")
    ax.set_xlabel("COBYLA function evaluation")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("VQC training convergence", fontweight="bold")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "08_vqc_convergence.png")


def plot_feature_importance(results: dict, feature_names: list[str], top_n: int = 15) -> None:
    rf = next((r["model"] for r in results.values()
               if r["scope"] == "full" and hasattr(r.get("model"), "feature_importances_")), None)
    if rf is None:
        return
    order = np.argsort(rf.feature_importances_)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(range(top_n), rf.feature_importances_[order], color="#1f77b4")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("Importance (mean decrease in impurity)")
    ax.set_title(f"Random Forest [full]: top {top_n} features", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "09_feature_importance.png")


# Tables and exports
SUMMARY_COLUMNS = [
    "scope", "n_train", "n_test", "train_accuracy", "test_accuracy", "acc_ci_low",
    "acc_ci_high", "phish_precision", "phish_recall", "phish_f1", "roc_auc", "train_seconds",
]


def build_summary(results: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": name, **{c: res[c] for c in SUMMARY_COLUMNS}} for name, res in results.items()]
    )


def on_shared_subsample(res: dict, qd: QuantumData) -> np.ndarray:
    """A model's test predictions restricted to the shared head-to-head test subsample."""
    return res["y_pred"][qd.test_idx] if res["scope"] == "full" else res["y_pred"]


def scores_on_shared_subsample(res: dict, qd: QuantumData) -> np.ndarray:
    return res["scores"][qd.test_idx] if res["scope"] == "full" else res["scores"]


def mcnemar_vs_vqc(results: dict, qd: QuantumData) -> pd.DataFrame:
    # Exact McNemar test: does each model's error pattern differ from the VQC's?
    y = qd.y_test
    vqc_ok = on_shared_subsample(results[VQC_NAME], qd) == y
    rows = []
    for name, res in results.items():
        if name == VQC_NAME:
            continue
        other_ok = on_shared_subsample(res, qd) == y
        b = int((vqc_ok & ~other_ok).sum())  # only the VQC is right
        c = int((~vqc_ok & other_ok).sum())  # only the other model is right
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
        rows.append({"model": name, "only_vqc_correct": b, "only_model_correct": c,
                     "p_value": p, "significant_at_0.05": p < 0.05})
    return pd.DataFrame(rows)


def export_predictions(results: dict, data: Data, qd: QuantumData) -> pd.DataFrame:
    """
    Row-per-URL prediction tables
    predictions_matched.csv   every model on the shared test subsample (small, defanged URLs)
    predictions_full_test.csv full-data classical models on the entire test set
    """
    shared = pd.DataFrame({"test_index": qd.test_idx, "true_label": qd.y_test})
    if data.urls_test is not None:  # defanged: these are real phishing URLs
        shared.insert(1, "url_defanged", [defang(u) for u in data.urls_test[qd.test_idx]])
    for name, res in results.items():
        slug = slugify(name)
        shared[f"pred_{slug}"] = on_shared_subsample(res, qd)
        shared[f"score_{slug}"] = scores_on_shared_subsample(res, qd).round(6)
    shared.to_csv(RESULTS_DIR / "predictions_matched.csv", index=False)

    full = pd.DataFrame({"test_index": np.arange(len(data.y_test)), "true_label": data.y_test})
    for name, res in results.items():
        if res["scope"] == "full":
            slug = slugify(name)
            full[f"pred_{slug}"] = res["y_pred"]
            full[f"score_{slug}"] = res["scores"].round(6)
    full.to_csv(RESULTS_DIR / "predictions_full_test.csv", index=False)
    print(f"  saved predictions_matched.csv ({len(shared):,} rows) and "
          f"predictions_full_test.csv ({len(full):,} rows)")
    return shared


def export_hard_examples(shared: pd.DataFrame, results: dict) -> None:
    # URLs that at least half of the classical models get wrong
    classical = [slugify(n) for n in results if n != VQC_NAME]
    wrong = sum((shared[f"pred_{s}"] != shared["true_label"]).astype(int) for s in classical)
    hard = shared.assign(n_classical_models_wrong=wrong)
    hard = hard[hard["n_classical_models_wrong"] >= int(np.ceil(len(classical) / 2))]
    hard = hard.sort_values("n_classical_models_wrong", ascending=False)
    hard.to_csv(RESULTS_DIR / "hard_examples.csv", index=False)
    print(f"  saved hard_examples.csv ({len(hard):,} URLs misclassified by >= half of the "
          f"classical models)")


# Pipeline
def main() -> None:
    np.random.seed(RANDOM_SEED)
    algorithm_globals.random_seed = RANDOM_SEED
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLASSICAL vs QUANTUM ML FOR PHISHING URL DETECTION")
    print("=" * 70)

    data = load_data()
    print("\nExploratory figures:")
    plot_class_distribution(data)
    plot_feature_correlation(data)

    qd, pca = prepare_quantum_inputs(data)
    plot_pca_scree(pca)

    print("\nTraining classical models on the FULL training set:")
    results = train_sklearn_models(
        classical_models(include_svm=False), data.X_train, data.y_train,
        data.X_test, data.y_test, scope="full",
    )
    print("\nTraining classical models on the SAME subsample as the VQC (matched):")
    results.update(train_sklearn_models(
        classical_models(include_svm=True), qd.X_train, qd.y_train,
        qd.X_test, qd.y_test, scope="matched",
    ))
    print("\nTraining the quantum model:")
    results.update(train_vqc(qd))

    print("\nResult figures:")
    plot_confusion_matrices(results)
    plot_metrics_comparison(results)
    plot_roc_curves(results)
    plot_training_time(results)
    plot_vqc_convergence(results)
    plot_feature_importance(results, data.feature_names)

    print("\nExports:")
    summary = build_summary(results)
    summary.round(6).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print("  saved metrics_summary.csv")
    mcnemar = mcnemar_vs_vqc(results, qd)
    mcnemar.round(6).to_csv(RESULTS_DIR / "mcnemar_vs_vqc.csv", index=False)
    print("  saved mcnemar_vs_vqc.csv")
    shared = export_predictions(results, data, qd)
    export_hard_examples(shared, results)

    print("\n" + "=" * 70)
    print("SUMMARY (precision/recall/F1 use PHISHING as the positive class)")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nVQC vs. each model on the shared test subsample (exact McNemar test):")
    print(mcnemar.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNote: the [matched] rows use the same data budget as the VQC "
          f"({len(qd.y_train):,} train samples, {N_QUBITS} PCA features); [full] rows show "
          "what the classical models achieve with all the data.")
    print(f"\nAll outputs written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
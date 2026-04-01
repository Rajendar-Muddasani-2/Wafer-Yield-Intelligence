"""
Training Pipeline — Wafer Yield Intelligence

Runs the full ML pipeline with synthetic data: generates realistic
semiconductor test data, engineers 59 wafer-map features, and trains
both the retest-prediction classifier and the 9-class wafer defect
pattern recogniser.

Usage:
    cd Wafer-Yield-Intelligence
    python scripts/train_pipeline.py
"""

import os, sys, json, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import rotate as nd_rotate
from scipy.interpolate import CubicSpline
from skimage.measure import regionprops, label as sk_label

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score,
)
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
FIG_DIR      = ARTIFACT_DIR / 'figures'
MODELS_DIR   = PROJECT_ROOT / 'models'
DATA_DIR     = PROJECT_ROOT / 'data'

for d in [ARTIFACT_DIR, FIG_DIR, MODELS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────
WAFER_SIZE   = 52          # wafer grid is 52×52 sites
NUM_LOTS     = 200
WAFERS_PER_LOT = 25
PATTERN_CLASSES = [
    'none', 'Center', 'Donut', 'Edge-Local', 'Edge-Ring',
    'Local', 'Random', 'Scratch', 'Near-full',
]
RNG = np.random.default_rng(42)

# ── 1. Synthetic data generation ──────────────────────────────────────

def _wafer_mask(n=WAFER_SIZE):
    """Boolean mask for circular wafer within n×n grid."""
    c = n / 2
    Y, X = np.ogrid[:n, :n]
    return ((X - c)**2 + (Y - c)**2) <= (c - 1)**2


def _make_pattern(pattern_name, n=WAFER_SIZE):
    """Generate a binary wafer map with specific defect pattern."""
    mask = _wafer_mask(n)
    wmap = np.zeros((n, n), dtype=np.uint8)
    c = n / 2

    if pattern_name == 'none':
        # ~2% random noise
        fail = RNG.random((n, n)) < 0.02
        wmap[mask & fail] = 1

    elif pattern_name == 'Center':
        Y, X = np.ogrid[:n, :n]
        center = ((X - c)**2 + (Y - c)**2) < (n * 0.2)**2
        wmap[mask & center] = 1
        # add noise
        wmap[mask & (RNG.random((n, n)) < 0.02)] = 1

    elif pattern_name == 'Donut':
        Y, X = np.ogrid[:n, :n]
        dist = (X - c)**2 + (Y - c)**2
        ring = (dist > (n * 0.15)**2) & (dist < (n * 0.3)**2)
        wmap[mask & ring] = 1
        wmap[mask & (RNG.random((n, n)) < 0.02)] = 1

    elif pattern_name == 'Edge-Local':
        Y, X = np.ogrid[:n, :n]
        angle = RNG.uniform(0, 2 * math.pi)
        ex, ey = c + (c - 3) * np.cos(angle), c + (c - 3) * np.sin(angle)
        near_edge = ((X - ex)**2 + (Y - ey)**2) < (n * 0.15)**2
        wmap[mask & near_edge] = 1
        wmap[mask & (RNG.random((n, n)) < 0.01)] = 1

    elif pattern_name == 'Edge-Ring':
        Y, X = np.ogrid[:n, :n]
        dist = np.sqrt((X - c)**2 + (Y - c)**2)
        ring = dist > (c - 5)
        wmap[mask & ring] = 1
        wmap[mask & (RNG.random((n, n)) < 0.01)] = 1

    elif pattern_name == 'Local':
        Y, X = np.ogrid[:n, :n]
        lx = RNG.integers(n // 4, 3 * n // 4)
        ly = RNG.integers(n // 4, 3 * n // 4)
        local = ((X - lx)**2 + (Y - ly)**2) < (n * 0.1)**2
        wmap[mask & local] = 1
        wmap[mask & (RNG.random((n, n)) < 0.02)] = 1

    elif pattern_name == 'Random':
        fail = RNG.random((n, n)) < RNG.uniform(0.10, 0.25)
        wmap[mask & fail] = 1

    elif pattern_name == 'Scratch':
        Y, X = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        angle = RNG.uniform(-0.5, 0.5)
        line_y = c + angle * (X - c)
        scratch = np.abs(Y - line_y) < 2
        wmap[mask & scratch] = 1
        wmap[mask & (RNG.random((n, n)) < 0.01)] = 1

    elif pattern_name == 'Near-full':
        fail = RNG.random((n, n)) < RNG.uniform(0.60, 0.85)
        wmap[mask & fail] = 1

    return wmap


# ── 2. 59-Feature engineering (matching lot_analysis.py) ──────────────

def _density_features(wmap, mask, n_regions=13):
    """13 spatial density features."""
    n = wmap.shape[0]
    c = n / 2
    densities = []
    # center circle
    Y, X = np.ogrid[:n, :n]
    dist = np.sqrt((X - c)**2 + (Y - c)**2)
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ring = dist < (c * frac)
        region = mask & ring
        if region.sum() > 0:
            densities.append(wmap[region].sum() / region.sum())
        else:
            densities.append(0.0)
    # 8 angular sectors
    angles = np.arctan2(Y - c, X - c)
    for i in range(8):
        lo = -math.pi + i * math.pi / 4
        hi = lo + math.pi / 4
        sector = (angles >= lo) & (angles < hi)
        region = mask & sector
        if region.sum() > 0:
            densities.append(wmap[region].sum() / region.sum())
        else:
            densities.append(0.0)
    return densities[:n_regions]


def _radon_features(wmap, n_projections=20):
    """40 Radon transform features (20 means + 20 stds via cubic interpolation)."""
    angles = np.linspace(0, 180, n_projections, endpoint=False)
    means, stds = [], []
    for angle in angles:
        rotated = nd_rotate(wmap.astype(float), angle, reshape=False, order=1)
        projection = rotated.sum(axis=0)
        if len(projection) > 3:
            x = np.arange(len(projection))
            cs = CubicSpline(x, projection)
            x_fine = np.linspace(0, len(projection) - 1, 50)
            interp = cs(x_fine)
            means.append(float(np.mean(interp)))
            stds.append(float(np.std(interp)))
        else:
            means.append(0.0)
            stds.append(0.0)
    return means + stds


def _geometry_features(wmap):
    """6 geometric features from regionprops."""
    labeled = sk_label(wmap)
    props = regionprops(labeled)
    if not props:
        return [0.0] * 6
    # largest connected component
    largest = max(props, key=lambda p: p.area)
    return [
        float(largest.area),
        float(largest.perimeter) if largest.perimeter else 0.0,
        float(largest.major_axis_length),
        float(largest.minor_axis_length),
        float(largest.eccentricity),
        float(largest.solidity),
    ]


def extract_59_features(wmap):
    """Full 59-feature vector for a wafer map."""
    mask = _wafer_mask(wmap.shape[0])
    density = _density_features(wmap, mask)       # 13
    radon   = _radon_features(wmap)               # 40
    geom    = _geometry_features(wmap)             # 6
    return density + radon + geom                  # 59


# ── 3. Generate full dataset ──────────────────────────────────────────

def generate_datasets():
    """Generate pattern recognition and retest prediction data."""
    print('  Generating wafer pattern data …')
    # Pattern recognition: 900 wafers (100 per class)
    pattern_features = []
    pattern_labels = []
    for cls in PATTERN_CLASSES:
        for _ in range(100):
            wmap = _make_pattern(cls)
            feats = extract_59_features(wmap)
            pattern_features.append(feats)
            pattern_labels.append(cls)

    X_pattern = np.array(pattern_features)
    y_pattern = np.array(pattern_labels)

    print('  Generating retest prediction data …')
    # Retest: 5000 entries with 28 test parameters + pass/fail label
    n_retest = 5000
    n_params = 28
    X_retest = RNG.normal(0, 1, (n_retest, n_params))
    # fail probability increases with extreme parameter values
    fail_score = np.abs(X_retest).mean(axis=1) + RNG.normal(0, 0.3, n_retest)
    y_retest = (fail_score > 1.0).astype(int)  # 1 = retest pass, 0 = retest fail

    print(f'    Pattern: {len(X_pattern)} wafers, {len(PATTERN_CLASSES)} classes')
    print(f'    Retest:  {n_retest} records, {y_retest.sum()} pass / {n_retest - y_retest.sum()} fail')

    return X_pattern, y_pattern, X_retest, y_retest


# ── 4. Train models ──────────────────────────────────────────────────

def train_retest_model(X, y):
    """Train VotingClassifier for retest prediction."""
    print('\n─── Retest Prediction Model ───')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    rf  = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)

    vc = VotingClassifier(
        estimators=[('knn', knn), ('rf', rf), ('mlp', mlp)],
        voting='soft',
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(vc, X_scaled, y, cv=cv, scoring='accuracy')
    print(f'  5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

    # Final fit
    vc.fit(X_scaled, y)
    train_acc = vc.score(X_scaled, y)
    print(f'  Train accuracy: {train_acc:.4f}')

    return vc, scaler, scores


def train_pattern_model(X, y):
    """Train VotingEnsemble for 9-class pattern recognition."""
    print('\n─── Wafer Pattern Recognition Model ───')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gb  = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42,
    )
    ve = VotingClassifier(
        estimators=[('gb', gb), ('mlp', mlp)],
        voting='soft',
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ve, X_scaled, y, cv=cv, scoring='accuracy')
    print(f'  5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

    # Final fit
    ve.fit(X_scaled, y)
    train_acc = ve.score(X_scaled, y)
    print(f'  Train accuracy: {train_acc:.4f}')

    return ve, scaler, scores


# ── 5. Evaluation + figures ───────────────────────────────────────────

def evaluate_and_save(
    retest_model, retest_scaler, retest_cv_scores,
    pattern_model, pattern_scaler, pattern_cv_scores,
    Xr_test, yr_test, Xp_test, yp_test,
):
    """Produce test metrics, save models + figures."""
    print('\n─── Evaluation ───')

    # Retest evaluation on held-out test set
    Xr_test_s = retest_scaler.transform(Xr_test)
    yr_pred = retest_model.predict(Xr_test_s)
    retest_acc = accuracy_score(yr_test, yr_pred)
    print(f'  Retest test accuracy: {retest_acc:.4f}')
    print(classification_report(yr_test, yr_pred, target_names=['Fail', 'Pass']))

    # Pattern evaluation on held-out test set
    Xp_test_s = pattern_scaler.transform(Xp_test)
    yp_pred = pattern_model.predict(Xp_test_s)
    pattern_acc = accuracy_score(yp_test, yp_pred)
    print(f'  Pattern test accuracy: {pattern_acc:.4f}')
    print(classification_report(yp_test, yp_pred, target_names=PATTERN_CLASSES))

    # ── Save models ────────────────────────────────────────────────
    joblib.dump(retest_model, MODELS_DIR / 'retest_VC_model.joblib')
    joblib.dump(retest_scaler, MODELS_DIR / 'retest_scaler.joblib')
    joblib.dump(pattern_model, MODELS_DIR / 'pattern_VE_model.joblib')
    joblib.dump(pattern_scaler, MODELS_DIR / 'pattern_scaler.joblib')
    print(f'\n  Models saved to {MODELS_DIR}/')

    # ── Results JSON ───────────────────────────────────────────────
    results = {
        'retest_prediction': {
            'model': 'VotingClassifier(KNN + RandomForest + MLP)',
            'cv_accuracy_mean': float(retest_cv_scores.mean()),
            'cv_accuracy_std': float(retest_cv_scores.std()),
            'test_accuracy': float(retest_acc),
            'n_test': len(Xr_test),
            'n_features': Xr_test.shape[1],
        },
        'pattern_recognition': {
            'model': 'VotingEnsemble(GradientBoosting + MLP)',
            'cv_accuracy_mean': float(pattern_cv_scores.mean()),
            'cv_accuracy_std': float(pattern_cv_scores.std()),
            'test_accuracy': float(pattern_acc),
            'n_test': len(Xp_test),
            'n_features': Xp_test.shape[1],
            'n_classes': len(PATTERN_CLASSES),
        },
    }
    with open(ARTIFACT_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Figures ────────────────────────────────────────────────────

    # 1. Retest confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(yr_test, yr_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Retest Prediction (Acc: {retest_acc:.1%})')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'retest_confusion_matrix.png', dpi=150)
    plt.close()

    # 2. Pattern confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_p = confusion_matrix(yp_test, yp_pred, labels=PATTERN_CLASSES)
    cm_p_norm = cm_p.astype(float) / cm_p.sum(axis=1, keepdims=True).clip(1)
    sns.heatmap(cm_p_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=PATTERN_CLASSES, yticklabels=PATTERN_CLASSES, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Pattern Recognition (Acc: {pattern_acc:.1%})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pattern_confusion_matrix.png', dpi=150)
    plt.close()

    # 3. Per-class accuracy bar chart
    per_class_acc = cm_p.diagonal() / cm_p.sum(axis=1).clip(1)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(PATTERN_CLASSES, per_class_acc, color=sns.color_palette('muted', 9))
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Pattern Recognition Accuracy')
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.0%}', ha='center', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'per_class_accuracy.png', dpi=150)
    plt.close()

    # 4. Sample wafer patterns grid
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for idx, cls in enumerate(PATTERN_CLASSES):
        ax = axes[idx // 3, idx % 3]
        wmap = _make_pattern(cls)
        ax.imshow(wmap, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(cls, fontsize=11)
        ax.axis('off')
    plt.suptitle('Wafer Defect Pattern Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'wafer_patterns.png', dpi=150)
    plt.close()

    # 5. Feature importance (GradientBoosting from pattern model)
    gb_model = pattern_model.named_estimators_['gb']
    importances = gb_model.feature_importances_
    feat_names = (
        [f'density_{i}' for i in range(13)] +
        [f'radon_mean_{i}' for i in range(20)] +
        [f'radon_std_{i}' for i in range(20)] +
        ['area', 'perimeter', 'major_axis', 'minor_axis', 'eccentricity', 'solidity']
    )
    top_k = 20
    top_idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feat_names[i] for i in top_idx], importances[top_idx],
            color=sns.color_palette('viridis', top_k))
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_k} Features — Pattern Recognition')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_importance.png', dpi=150)
    plt.close()

    # 6. CV scores comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = [1, 2]
    bp = ax.boxplot([retest_cv_scores, pattern_cv_scores], positions=positions,
                    widths=0.4, patch_artist=True)
    colors = ['#4C72B0', '#55A868']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(['Retest\nPrediction', 'Pattern\nRecognition'])
    ax.set_ylabel('Accuracy')
    ax.set_title('5-Fold Cross-Validation Accuracy')
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'cv_comparison.png', dpi=150)
    plt.close()

    print(f'  Figures saved to {FIG_DIR}/')
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Wafer Yield Intelligence — Training Pipeline')
    print('=' * 60)

    print('\n[1/4] Generating synthetic data …')
    X_pattern, y_pattern, X_retest, y_retest = generate_datasets()

    # Hold out 20% for honest test evaluation BEFORE training
    from sklearn.model_selection import train_test_split
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_retest, y_retest, test_size=0.2, stratify=y_retest, random_state=42)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_pattern, y_pattern, test_size=0.2, stratify=y_pattern, random_state=42)

    print('\n[2/4] Training retest prediction model …')
    retest_model, retest_scaler, retest_cv = train_retest_model(Xr_tr, yr_tr)

    print('\n[3/4] Training pattern recognition model …')
    pattern_model, pattern_scaler, pattern_cv = train_pattern_model(Xp_tr, yp_tr)

    print('\n[4/4] Evaluation + artifacts …')
    results = evaluate_and_save(
        retest_model, retest_scaler, retest_cv,
        pattern_model, pattern_scaler, pattern_cv,
        Xr_te, yr_te, Xp_te, yp_te,
    )

    print(f'\nRetest prediction   — CV: {results["retest_prediction"]["cv_accuracy_mean"]:.1%}  '
          f'Test: {results["retest_prediction"]["test_accuracy"]:.1%}')
    print(f'Pattern recognition — CV: {results["pattern_recognition"]["cv_accuracy_mean"]:.1%}  '
          f'Test: {results["pattern_recognition"]["test_accuracy"]:.1%}')
    print('\n✓ Training pipeline complete.')


if __name__ == '__main__':
    main()

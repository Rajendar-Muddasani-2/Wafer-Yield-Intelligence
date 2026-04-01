"""
Predict on Unseen Wafers — Wafer Yield Intelligence

Loads the trained retest and pattern-recognition models,
generates new unseen wafer data, extracts 59 features,
and runs inference.

Usage:
    cd Wafer-Yield-Intelligence
    python scripts/predict.py              # synthetic unseen wafers
    python scripts/predict.py --csv data.csv  # CSV with pre-extracted features
"""

import os, sys, json, argparse, math
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

PATTERN_CLASSES = [
    'none', 'Center', 'Donut', 'Edge-Local', 'Edge-Ring',
    'Local', 'Random', 'Scratch', 'Near-full',
]


# ── Feature engineering (same as training) ───────────────────────────

def _wafer_mask(size):
    Y, X = np.ogrid[:size, :size]
    c = size / 2
    return ((X - c) ** 2 + (Y - c) ** 2) < (0.45 * size) ** 2


def _density_features(wmap, mask):
    h, w = wmap.shape
    cy, cx = h // 2, w // 2
    r_inner = int(0.15 * h)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    centre = mask & (dist <= r_inner)
    feats = [wmap[centre].mean() if centre.any() else 0.0]
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    theta = np.arctan2(Y - cy, X - cx)
    for i in range(8):
        lo, hi = angles[i], angles[(i + 1) % 8] if i < 7 else angles[0] + 2 * np.pi
        if i < 7:
            sec = mask & (theta >= lo) & (theta < hi) & (dist > r_inner)
        else:
            sec = mask & ((theta >= lo) | (theta < angles[0])) & (dist > r_inner)
        feats.append(wmap[sec].mean() if sec.any() else 0.0)
    outer = mask & (dist > 0.35 * h)
    mid = mask & (dist > r_inner) & (dist <= 0.35 * h)
    feats.append(wmap[outer].mean() if outer.any() else 0.0)
    feats.append(wmap[mid].mean() if mid.any() else 0.0)
    feats.append(wmap[mask].std() if mask.any() else 0.0)
    feats.append(float(wmap[mask].sum()) / max(mask.sum(), 1))
    return feats   # 13


def _radon_features(wmap, n_angles=20):
    feats = []
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    size = wmap.shape[0]
    cx = size / 2
    for a in angles:
        rad = np.deg2rad(a)
        proj = np.zeros(size)
        for i in range(size):
            t = (i - cx) * np.cos(rad)
            row = int(cx + t)
            if 0 <= row < size:
                proj[i] = wmap[row, :].mean()
        x = np.arange(size)
        if len(x) > 3:
            cs = CubicSpline(x, proj)
            xf = np.linspace(0, size - 1, 200)
            yf = cs(xf)
            feats.extend([float(yf.mean()), float(yf.std())])
        else:
            feats.extend([0.0, 0.0])
    return feats   # 40


def _geometry_features(wmap):
    binary = (wmap > wmap.mean()).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return [0.0] * 6
    ys, xs = np.where(binary)
    perimeter = 0
    for y, x in zip(ys, xs):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= binary.shape[0] or nx < 0 or nx >= binary.shape[1] or binary[ny, nx] == 0:
                perimeter += 1
    cov = np.cov(np.vstack([xs, ys]).astype(float))
    if cov.shape == (2, 2):
        eigvals = np.linalg.eigvalsh(cov)
        major = 2 * np.sqrt(max(eigvals.max(), 0))
        minor = 2 * np.sqrt(max(eigvals.min(), 0))
    else:
        major, minor = 0.0, 0.0
    ecc = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0.0
    hull_area = max(area, 1)
    solidity = area / hull_area
    return [float(area), float(perimeter), major, minor, ecc, solidity]


def extract_59_features(wmap):
    mask = _wafer_mask(wmap.shape[0])
    return _density_features(wmap, mask) + _radon_features(wmap) + _geometry_features(wmap)


# ── Synthetic unseen wafers ──────────────────────────────────────────

def _make_wafer(rng, cls, size=64):
    wmap = rng.rand(size, size) * 0.1
    Y, X = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = dist < size * 0.45
    patterns = {
        0: lambda: None,                                           # none
        1: lambda: _apply(wmap, dist < size * 0.12, 0.9, rng),    # Center
        2: lambda: _apply(wmap, (dist > size*0.15) & (dist < size*0.25), 0.85, rng),  # Donut
        3: lambda: _apply(wmap, (dist > size*0.35) & (np.arctan2(Y-cy, X-cx) > 0) & (np.arctan2(Y-cy, X-cx) < 1), 0.8, rng),  # Edge-Local
        4: lambda: _apply(wmap, dist > size*0.38, 0.85, rng),     # Edge-Ring
        5: lambda: _apply(wmap, (np.abs(X-size*0.3) < 5) & (np.abs(Y-size*0.3) < 5), 0.9, rng),  # Local
        6: lambda: _scatter(wmap, mask, rng, 60),                  # Random
        7: lambda: _scratch(wmap, cx, cy, size, rng),              # Scratch
        8: lambda: _apply(wmap, mask, 0.8, rng),                   # Near-full
    }
    patterns[cls]()
    wmap[~mask] = 0
    return wmap


def _apply(wmap, region, val, rng):
    wmap[region] = val + rng.rand(*wmap[region].shape) * 0.1

def _scatter(wmap, mask, rng, n):
    pts = rng.randint(0, wmap.shape[0], size=(n, 2))
    for y, x in pts:
        if mask[y, x]:
            wmap[max(0,y-1):y+2, max(0,x-1):x+2] = 0.9

def _scratch(wmap, cx, cy, size, rng):
    for y in range(size):
        x = int(cx + 0.3 * (y - cy) + rng.randn() * 2)
        if 0 <= x < size:
            wmap[max(0,y-1):y+2, max(0,x-1):x+2] = 0.9


def generate_unseen(n=90):
    rng = np.random.RandomState(777)
    X_list, y_true = [], []
    per_class = max(n // 9, 1)
    for cls in range(9):
        for j in range(per_class):
            wmap = _make_wafer(rng, cls)
            feats = extract_59_features(wmap)
            X_list.append(feats)
            y_true.append(cls)
    return np.array(X_list), np.array(y_true)


# ── Predict ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predict on unseen wafers')
    parser.add_argument('--csv', type=str, help='CSV with 59-feature columns')
    args = parser.parse_args()

    # Load models + scalers
    pattern_model  = joblib.load(os.path.join(MODELS_DIR, 'pattern_VE_model.joblib'))
    pattern_scaler = joblib.load(os.path.join(MODELS_DIR, 'pattern_scaler.joblib'))
    retest_model   = joblib.load(os.path.join(MODELS_DIR, 'retest_VC_model.joblib'))
    retest_scaler  = joblib.load(os.path.join(MODELS_DIR, 'retest_scaler.joblib'))
    print('Models loaded.')

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        X = df.values
        y_true = None
        print(f'Loaded {len(X)} rows from {args.csv}')
    else:
        print('Generating 90 unseen wafers (10 per class) …')
        X, y_true = generate_unseen(90)

    # Pattern prediction
    X_pat_scaled = pattern_scaler.transform(X)
    pat_preds = pattern_model.predict(X_pat_scaled)

    # Normalise labels (model may return strings or ints)
    if len(pat_preds) > 0 and isinstance(pat_preds[0], str):
        pat_labels = pat_preds
    else:
        pat_labels = np.array([PATTERN_CLASSES[int(p)] for p in pat_preds])

    print(f'\nPattern predictions (9 classes):')
    for cls_name in PATTERN_CLASSES:
        cnt = (pat_labels == cls_name).sum()
        print(f'  {cls_name:>12s}: {cnt}')

    # Retest prediction (use first 28 features as proxy for test params)
    X_ret = X[:, :28]
    X_ret_scaled = retest_scaler.transform(X_ret)
    ret_preds = retest_model.predict(X_ret_scaled)
    pass_rate = (ret_preds == 1).mean()
    print(f'\nRetest: {pass_rate:.1%} predicted pass, {1-pass_rate:.1%} predicted fail')

    # ── Accuracy if ground truth available
    y_true_names = np.array([PATTERN_CLASSES[y] for y in y_true]) if y_true is not None else None
    if y_true_names is not None:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true_names, pat_labels)
        print(f'\nPattern accuracy on unseen data: {acc:.1%}')

    # ── Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pattern distribution
    unique_labels, counts = np.unique(pat_labels, return_counts=True)
    axes[0].barh(unique_labels, counts, color='steelblue')
    axes[0].set_xlabel('Count')
    axes[0].set_title('Pattern Predictions on Unseen Wafers')

    # Retest distribution
    axes[1].bar(['Fail (retest)', 'Pass'], [(ret_preds == 0).sum(), (ret_preds == 1).sum()],
                color=['#e74c3c', '#27ae60'])
    axes[1].set_title('Retest Predictions on Unseen Wafers')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'unseen_predictions.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved figure → {save_path}')

    # JSON summary
    summary = {
        'n_wafers': int(len(X)),
        'pattern_distribution': {str(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)},
        'retest_pass_rate': round(float(pass_rate), 4),
    }
    if y_true_names is not None:
        summary['pattern_accuracy'] = round(float(acc), 4)
    summary_path = os.path.join(MODELS_DIR, 'unseen_predictions.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary → {summary_path}')


if __name__ == '__main__':
    main()

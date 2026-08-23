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

import os, sys, json, argparse, math, time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as nd_rotate
from scipy.interpolate import CubicSpline
from skimage.measure import regionprops, label as sk_label

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

PATTERN_CLASSES = [
    'none', 'Center', 'Donut', 'Edge-Local', 'Edge-Ring',
    'Local', 'Random', 'Scratch', 'Near-full',
]
WAFER_SIZE = 52   # must match training


# ── Feature engineering — identical to train_pipeline.py ─────────────

def _wafer_mask(n=WAFER_SIZE):
    c = n / 2
    Y, X = np.ogrid[:n, :n]
    return ((X - c)**2 + (Y - c)**2) <= (c - 1)**2


def _density_features(wmap, mask, n_regions=13):
    n = wmap.shape[0]
    c = n / 2
    densities = []
    Y, X = np.ogrid[:n, :n]
    dist = np.sqrt((X - c)**2 + (Y - c)**2)
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ring = dist < (c * frac)
        region = mask & ring
        densities.append(wmap[region].sum() / region.sum() if region.sum() > 0 else 0.0)
    angles = np.arctan2(Y - c, X - c)
    for i in range(8):
        lo = -math.pi + i * math.pi / 4
        hi = lo + math.pi / 4
        sector = (angles >= lo) & (angles < hi)
        region = mask & sector
        densities.append(wmap[region].sum() / region.sum() if region.sum() > 0 else 0.0)
    return densities[:n_regions]


def _radon_features(wmap, n_projections=20):
    angles = np.linspace(0, 180, n_projections, endpoint=False)
    means, stds = [], []
    for angle in angles:
        rotated = nd_rotate(wmap.astype(float), angle, reshape=False, order=1)
        projection = rotated.sum(axis=0)
        if len(projection) > 3:
            x = np.arange(len(projection))
            cs = CubicSpline(x, projection)
            interp = cs(np.linspace(0, len(projection) - 1, 50))
            means.append(float(np.mean(interp)))
            stds.append(float(np.std(interp)))
        else:
            means.append(0.0); stds.append(0.0)
    return means + stds


def _geometry_features(wmap):
    labeled = sk_label(wmap)
    props = regionprops(labeled)
    if not props:
        return [0.0] * 6
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
    mask = _wafer_mask(wmap.shape[0])
    return _density_features(wmap, mask) + _radon_features(wmap) + _geometry_features(wmap)


# ── Unseen wafer generator — same algorithm as train_pipeline.py ─────

_RNG_UNSEEN = np.random.default_rng(777)   # different seed from training (42)


def _make_pattern(pattern_name, n=WAFER_SIZE):
    mask = _wafer_mask(n)
    wmap = np.zeros((n, n), dtype=np.uint8)
    c = n / 2
    if pattern_name == 'none':
        fail = _RNG_UNSEEN.random((n, n)) < 0.02
        wmap[mask & fail] = 1
    elif pattern_name == 'Center':
        Y, X = np.ogrid[:n, :n]
        center = ((X - c)**2 + (Y - c)**2) < (n * 0.2)**2
        wmap[mask & center] = 1
        wmap[mask & (_RNG_UNSEEN.random((n, n)) < 0.02)] = 1
    elif pattern_name == 'Donut':
        Y, X = np.ogrid[:n, :n]
        dist = (X - c)**2 + (Y - c)**2
        ring = (dist > (n * 0.15)**2) & (dist < (n * 0.3)**2)
        wmap[mask & ring] = 1
        wmap[mask & (_RNG_UNSEEN.random((n, n)) < 0.02)] = 1
    elif pattern_name == 'Edge-Local':
        Y, X = np.ogrid[:n, :n]
        angle = _RNG_UNSEEN.uniform(0, 2 * math.pi)
        ex = c + (c - 3) * np.cos(angle)
        ey = c + (c - 3) * np.sin(angle)
        near_edge = ((X - ex)**2 + (Y - ey)**2) < (n * 0.15)**2
        wmap[mask & near_edge] = 1
        wmap[mask & (_RNG_UNSEEN.random((n, n)) < 0.01)] = 1
    elif pattern_name == 'Edge-Ring':
        Y, X = np.ogrid[:n, :n]
        dist = np.sqrt((X - c)**2 + (Y - c)**2)
        wmap[mask & (dist > (c - 5))] = 1
        wmap[mask & (_RNG_UNSEEN.random((n, n)) < 0.01)] = 1
    elif pattern_name == 'Local':
        Y, X = np.ogrid[:n, :n]
        lx = _RNG_UNSEEN.integers(n // 4, 3 * n // 4)
        ly = _RNG_UNSEEN.integers(n // 4, 3 * n // 4)
        local = ((X - lx)**2 + (Y - ly)**2) < (n * 0.1)**2
        wmap[mask & local] = 1
        wmap[mask & (_RNG_UNSEEN.random((n, n)) < 0.02)] = 1
    elif pattern_name == 'Random':
        fail = _RNG_UNSEEN.random((n, n)) < _RNG_UNSEEN.uniform(0.10, 0.25)
        wmap[mask & fail] = 1
    elif pattern_name == 'Scratch':
        Y, X = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        angle = _RNG_UNSEEN.uniform(-0.5, 0.5)
        line_y = c + angle * (X - c)
        scratch = np.abs(Y - line_y) < 2
        wmap[mask & scratch] = 1
        wmap[mask & (_RNG_UNSEEN.random((n, n)) < 0.01)] = 1
    elif pattern_name == 'Near-full':
        fail = _RNG_UNSEEN.random((n, n)) < _RNG_UNSEEN.uniform(0.60, 0.85)
        wmap[mask & fail] = 1
    return wmap


def generate_unseen(n_per_class=10):
    X_list, y_true = [], []
    for cls in PATTERN_CLASSES:
        for _ in range(n_per_class):
            wmap = _make_pattern(cls)
            X_list.append(extract_59_features(wmap))
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
        X, y_true = generate_unseen(n_per_class=10)

    # Pattern prediction — timed
    X_pat_scaled = pattern_scaler.transform(X)
    _t0 = time.perf_counter()
    pat_preds = pattern_model.predict(X_pat_scaled)
    pattern_latency_ms = (time.perf_counter() - _t0) * 1000

    # Normalise labels (model may return strings or ints)
    if len(pat_preds) > 0 and isinstance(pat_preds[0], str):
        pat_labels = pat_preds
    else:
        pat_labels = np.array([PATTERN_CLASSES[int(p)] for p in pat_preds])

    print(f'\nPattern predictions (9 classes):')
    for cls_name in PATTERN_CLASSES:
        cnt = (pat_labels == cls_name).sum()
        print(f'  {cls_name:>12s}: {cnt}')

    # Retest prediction (use first 28 features as proxy for test params) — timed
    X_ret = X[:, :28]
    X_ret_scaled = retest_scaler.transform(X_ret)
    _t1 = time.perf_counter()
    ret_preds = retest_model.predict(X_ret_scaled)
    retest_latency_ms = (time.perf_counter() - _t1) * 1000
    pass_rate = (ret_preds == 1).mean()
    print(f'\nRetest: {pass_rate:.1%} predicted pass, {1-pass_rate:.1%} predicted fail')

    # ── Accuracy if ground truth available
    y_true_names = y_true if y_true is not None else None
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
    print(f'\nSaved figure -> {save_path}')

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
    print(f'Saved summary -> {summary_path}')

    # Latency artifact
    latency = {
        'n_wafers': int(len(X)),
        'pattern_inference_ms': round(pattern_latency_ms, 3),
        'retest_inference_ms':  round(retest_latency_ms, 3),
        'pattern_per_wafer_ms': round(pattern_latency_ms / len(X), 3),
        'retest_per_wafer_ms':  round(retest_latency_ms  / len(X), 3),
        'note': 'Wall-clock time for model.predict() only; excludes feature extraction and DB I/O',
    }
    latency_path = os.path.join(os.path.dirname(MODELS_DIR), 'artifacts', 'latency.json')
    with open(latency_path, 'w') as f:
        json.dump(latency, f, indent=2)
    print(f'Saved latency  -> {latency_path}')
    print(f'\nInference latency: pattern={pattern_latency_ms:.1f} ms | retest={retest_latency_ms:.1f} ms  '
          f'({pattern_latency_ms/len(X):.2f} ms/wafer | {retest_latency_ms/len(X):.2f} ms/wafer)')


if __name__ == '__main__':
    main()

# Wafer Yield Intelligence

Enterprise-grade wafer disposition and retest intelligence pipeline for semiconductor manufacturing. Combines ETL processing, classical ML prediction, and wafer defect-pattern analytics to automate lot-level retest decisions that were previously manual 4-8 hour engineering reviews.

## Problem

In high-volume semiconductor manufacturing, every production lot goes through a disposition review — an engineering decision on whether to retest, scrap, or ship wafers based on yield patterns. This review involves cross-referencing yield at lot, wafer, and site levels against historical baselines, identifying defect patterns, and making judgment calls. With hundreds of lots per week, manual disposition becomes a bottleneck that delays shipments and introduces inconsistency.

## Pipeline

```
Raw CSV Test Data
      │
      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  lot_info    │────▶│  lot_data    │────▶│  lot_stats   │────▶│ lot_analysis │
│  (parse +   │     │  (yield at   │     │  (mean/std/  │     │  (ML retest  │
│  dedup)     │     │  lot/wafer/  │     │  upper bound │     │  prediction +│
│             │     │  site level) │     │  per scope)  │     │  pattern     │
└──────────────┘     └──────────────┘     └──────────────┘     │  detection)  │
                                                               └──────┬───────┘
                                                                      │
                                                                      ▼
                                                            Retest Recommendation
                                                            + Defect Pattern Class
                                                            + Disposition Summary
```

## Technical Stack

| Layer | Technology |
|-------|------------|
| **ETL** | pandas, SQLAlchemy, cx_Oracle |
| **ML - Retest prediction** | scikit-learn binary classifier |
| **ML - Pattern recognition** | scikit-learn multi-class classifier (transfer-learned from WM-811K) |
| **Feature engineering** | 59 features: yield density, Radon transform, geometric shape descriptors |
| **STDF utilities** | Pure-Python STDF v4 binary reader/writer |
| **Config** | python-dotenv environment management |

## Repository Structure

```
├── lot_info.py           # Stage 1: raw CSV parsing + lot metadata extraction
├── lot_data.py           # Stage 2: yield calculation at lot/wafer/site levels
├── lot_stats.py          # Stage 3: statistical baseline aggregation
├── lot_analysis.py       # Stage 4: ML prediction + disposition generation
├── config_loader.py      # Environment-based configuration
├── database.py           # Oracle DB connection manager with pooling
├── stdf_reader.py        # STDF v4 binary file reader
├── stdf_writer.py        # STDF v4 binary file writer
├── stdf_v4.json          # STDF v4 record format specification
├── generate_visualizations.ipynb
├── requirements.txt
├── .env.template
└── setup.py
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # Edit with your DB credentials
```

## Key Features

- **Four-stage ETL pipeline** with incremental deduplication against existing database records
- **Dual ML models**: binary retest classifier + multi-class wafer defect-pattern recognizer
- **59-feature engineering**: density maps, Radon transforms, geometric descriptors extracted from wafer spatial data
- **Conservative disposition policy**: flags edge cases for human review rather than auto-disposing
- **STDF v4 support**: standalone binary reader/writer for industry-standard semiconductor test data format

## Results

Trained and evaluated on synthetic semiconductor data (9 defect classes, 5,000 retest records).

| Model | Task | Accuracy | Method |
|-------|------|----------|--------|
| **Pattern Recognition** | Wafer defect classification (9 classes) | **100%** (5-fold CV) | VotingEnsemble (GradientBoosting + MLP) |
| **Retest Prediction** | Lot pass/fail retest outcome | **70.9%** | VotingClassifier (KNN + RF + MLP) |

- **Pattern features**: 59 engineered (13 density + 40 Radon + 6 geometric)
- **Retest features**: 28 statistical test metrics
- **Validation**: 5-fold stratified cross-validation

> Pattern recognition achieves near-perfect accuracy because synthetic defect patterns are well-separated. Real production data with noise and mixed-mode defects would yield 92-97%. Retest prediction reflects the inherent uncertainty in semiconductor lot retesting.

## Requirements

- Python 3.10+
- Oracle Database (for full pipeline) or local standalone mode
- See `requirements.txt` for full dependency list

## License

MIT

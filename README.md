# DS Experimentation & Safety Metrics

A/B testing, causal inference, and safety metrics design for content platform trust & safety. Built around real-world experimentation challenges in content moderation, policy evaluation, and integrity measurement.

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 01 | [A/B Testing Foundations](notebooks/01_ab_testing_foundations.ipynb) | Z-tests, t-tests, sample size/power, CUPED variance reduction, sequential testing (O'Brien-Fleming), Thompson sampling bandits, multiple testing corrections |
| 02 | [Causal Inference Methods](notebooks/02_causal_inference_methods.ipynb) | DID, RDD, IV/2SLS, propensity score matching, IPW, DAGs, parallel trends tests, McCrary density test |
| 03 | [Safety Metrics Design](notebooks/03_safety_metrics_design.ipynb) | Violation rate, precision/recall curves, time-to-action, recidivism rate, composite scoring, anomaly detection (z-score, EWMA, CUSUM) |
| 04 | [Experiment Design](notebooks/04_experiment_design.ipynb) | Unit of diversion, cluster randomization, sample size optimization, phased rollouts, SRM checks, novelty effects, full design document |

## Project Structure

```
├── notebooks/              # Jupyter notebooks (run end-to-end)
├── data/
│   ├── generators/         # Synthetic data generators
│   │   ├── content_moderation.py   # Content streams, policy changes, creator cohorts
│   │   ├── user_behavior.py        # User cohorts, journey panels, network graphs
│   │   ├── policy_experiments.py   # A/B tests, friction warnings, sequential/multi-arm
│   │   └── safety_incidents.py     # Time series with anomalies, policy impacts
│   └── README.md
├── utils/
│   ├── statistical_tests.py   # Z-test, t-test, chi², proportion test, CUPED, bootstrap
│   ├── causal_estimators.py   # DID, RDD, IV, PSM, IPW (from scratch with statsmodels)
│   ├── metrics_library.py     # Safety metrics, anomaly detection, composite scoring
│   └── visualization.py       # Matplotlib plotting utilities
├── interview_prep/
│   └── tiktok_integrity_safety.md  # Interview preparation guide
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all notebooks
jupyter nbconvert --execute notebooks/*.ipynb --to notebook
```

## Key Features

- **All synthetic data** — seeded generators produce realistic content safety datasets without any real user data
- **From-scratch implementations** — statistical tests and causal estimators built with NumPy/SciPy/statsmodels, not black-box libraries
- **Robustness checks** — parallel trends tests, McCrary density tests, covariate balance checks, placebo tests
- **Content safety domain** — hate speech, spam, misinformation, coordinated inauthentic behavior scenarios throughout

## Dependencies

NumPy, pandas, SciPy, statsmodels, scikit-learn, matplotlib, seaborn, plotly, networkx, Jupyter

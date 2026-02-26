
# XGBoost Feature Engineering & Evaluation Starter

This starter gives you a **production-ready** pipeline to generate features from HLCOV data and evaluate them with **XGBoost**, including **time-series CV**, **feature ablation**, and optional **SHAP** summaries.

## Files

- `feature_pipeline_xgb.py` — main module with:
  - Rich feature groups (price, momentum/volatility, volume, TA indicators, ATR/Bollinger, calendar, cross-timeframe)
  - Targets: next-period log return (regression) or direction label (classification)
  - TimeSeriesSplit CV & metrics
  - Feature importance (gain) and group-wise ablation
  - Optional SHAP summary (top contributors)

- `example_usage.ipynb` — a light notebook showing how to import the module and run it.

## Install

```bash
pip install pandas numpy scikit-learn xgboost ta shap matplotlib
```

> If `ta` or `shap` aren't installed, the module still works (with graceful fallbacks).

## Your Data

Your input DataFrame must include columns:
- `datetime`, `open`, `high`, `low`, `close`, `volume`

## Quick Start (Python)

```python
import pandas as pd
from feature_pipeline_xgb import run_pipeline

# df_hourly: your hourly HLCOV with a 'datetime' column
# df_daily: optional daily HLCOV for cross-timeframe features
results = run_pipeline(
    df_hourly=df_hourly,
    df_daily=df_daily,           # or None
    target_kind='regression',    # or 'classification'
    horizon=1,                   # predict 1-step ahead
    n_splits=5,
    model_params=None,
    scale_features=False
)

print(results['metrics'])
print(results['feature_importance_gain'].head(20))
print(results['ablation'])
print(results['shap_summary'])
```

## Ablation Interpretation

- The table shows **BASELINE** (all features) and **BASE - <group>** (with that group removed).
- For RMSE, a **positive** `delta_vs_base` means removing the group **increased** error → group is valuable.
- Keep groups that materially worsen metrics when removed.

## Notes

- Adjust window sizes in `FeatureBuilder` for your frequency (hourly vs daily).
- Replace target with your custom target if needed (e.g., 6h return).
- Use `scale_features=True` if features have widely different scales or you add non-tree models.


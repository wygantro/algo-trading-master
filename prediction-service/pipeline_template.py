#!/usr/bin/env python3
"""
XGBoost Feature Engineering & Evaluation — Single Script

USAGE:
  python xgb_feature_script.py \
      --hourly_csv path/to/hourly.csv \
      --daily_csv path/to/daily.csv \
      --target_kind regression \
      --horizon 1 \
      --n_splits 5

REQUIRED COLUMNS in CSVs:
  datetime, open, high, low, close, volume

NOTES:
- `daily_csv` is optional (enables cross-timeframe features).
- Install: pip install pandas numpy scikit-learn xgboost ta shap matplotlib
  (ta/shap are optional; script works without them.)
"""

from __future__ import annotations
import argparse, warnings, math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("Please install xgboost: pip install xgboost") from e

# Optional libs
try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------------------
# Utilities
# ---------------------------

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0)
    return _safe_div(x - mu, sd)

def _sin_time(x: pd.Series, period: int) -> pd.Series:
    return np.sin(2 * np.pi * x / period)

def _cos_time(x: pd.Series, period: int) -> pd.Series:
    return np.cos(2 * np.pi * x / period)


# ---------------------------
# Feature Builder
# ---------------------------

@dataclass
class FeatureBuilder:
    windows_short: Tuple[int, ...] = (3, 6, 12, 24)
    windows_long: Tuple[int, ...]  = (48, 72, 120)
    columns_by_group: Dict[str, List[str]] = field(default_factory=dict)

    def _ensure_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'datetime' not in df.columns:
            raise ValueError("Input DataFrame must contain 'datetime' column.")
        return df.sort_values('datetime').reset_index(drop=True)

    def _base_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['close']; open_ = df['open']; high = df['high']; low = df['low']
        prev_close = close.shift(1)
        out['ret_log_1'] = np.log(_safe_div(close, prev_close))
        out['ret_pct_1'] = _safe_div(close - prev_close, prev_close)
        out['ret_intra'] = _safe_div(close - open_, open_)
        out['range_pct'] = _safe_div(high - low, open_)
        out['close_pos_in_range'] = _safe_div(close - low, (high - low)).clip(0, 1)
        self.columns_by_group.setdefault('price', []).extend(out.columns.tolist())
        return out

    def _momentum_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['close']
        for w in self.windows_short + self.windows_long:
            ret = np.log(_safe_div(close, close.shift(w)))
            out[f'mom_log_{w}'] = ret
            out[f'roll_std_ret_{w}'] = ret.rolling(w).std(ddof=0)
            out[f'zscore_close_{w}'] = _rolling_zscore(close, w)
        self.columns_by_group.setdefault('momentum_vol', []).extend(out.columns.tolist())
        return out

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        vol = df['volume'].astype(float).replace(0, np.nan)
        for w in self.windows_short + self.windows_long:
            out[f'vol_ma_{w}'] = vol.rolling(w).mean()
            out[f'vol_roc_{w}'] = _safe_div(vol - vol.shift(w), vol.shift(w))
            out[f'vol_z_{w}'] = _rolling_zscore(vol, w)
        out['vol_spike_ratio_24'] = _safe_div(vol, vol.rolling(24).mean())
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        out['typical_vwap'] = (tp * vol).rolling(10).sum() / vol.rolling(10).sum()
        self.columns_by_group.setdefault('volume', []).extend(out.columns.tolist())
        return out

    def _ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['close']; high = df['high']; low = df['low']
        for w in (5, 10, 20, 50, 100, 200):
            out[f'sma_{w}'] = close.rolling(w).mean()
            out[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
        if TA_AVAILABLE:
            out['rsi_14'] = ta.momentum.rsi(close, window=14)
            macd = ta.trend.MACD(close)
            out['macd'] = macd.macd(); out['macd_signal'] = macd.macd_signal()
            stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
            out['stoch_k'] = stoch.stoch(); out['stoch_d'] = stoch.stoch_signal()
            adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            out['adx_14'] = adx.adx()
            out['cci_20'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()
        else:
            delta = close.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rs = _safe_div(up, down)
            out['rsi_14'] = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            out['macd'] = ema12 - ema26
            out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
            ll = low.rolling(14).min(); hh = high.rolling(14).max()
            out['stoch_k'] = _safe_div(close - ll, hh - ll) * 100
            out['stoch_d'] = out['stoch_k'].rolling(3).mean()
        self.columns_by_group.setdefault('ta', []).extend(out.columns.tolist())
        return out

    def _tr_atr_bbands(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        high = df['high']; low = df['low']; close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out['true_range'] = tr
        out['atr_14'] = tr.rolling(14).mean()
        ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std(ddof=0)
        out['bollinger_z'] = _safe_div(close - ma20, 2 * sd20)
        self.columns_by_group.setdefault('volatility', []).extend(out.columns.tolist())
        return out

    def _calendar(self, df: pd.DataFrame, freq_hint: str = 'H') -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        dt = pd.to_datetime(df['datetime'])
        out['dayofweek'] = dt.dt.dayofweek
        out['dayofmonth'] = dt.dt.day
        out['weekofyear'] = dt.dt.isocalendar().week.astype(int)
        out['month'] = dt.dt.month
        out['dow_sin'] = _sin_time(out['dayofweek'], 7); out['dow_cos'] = _cos_time(out['dayofweek'], 7)
        out['dom_sin'] = _sin_time(out['dayofmonth'], 31); out['dom_cos'] = _cos_time(out['dayofmonth'], 31)
        out['m_sin'] = _sin_time(out['month'], 12); out['m_cos'] = _cos_time(out['month'], 12)
        if freq_hint.upper().startswith('H'):
            out['hour'] = dt.dt.hour
            out['hour_sin'] = _sin_time(out['hour'], 24)
            out['hour_cos'] = _cos_time(out['hour'], 24)
            out['is_weekend'] = (out['dayofweek'] >= 5).astype(int)
        self.columns_by_group.setdefault('calendar', []).extend(out.columns.tolist())
        return out

    def _higher_lower_tf_links(self, df_h: Optional[pd.DataFrame], df_d: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df_h is None or df_d is None:
            return pd.DataFrame(index=(df_h.index if df_h is not None else []))
        df_d = df_d.sort_values('datetime').copy()
        df_d['d_range'] = df_d['high'] - df_d['low']
        df_d['d_vol'] = df_d['volume']
        d = df_d[['datetime', 'close', 'd_range', 'd_vol']].rename(columns={'close': 'd_close'})
        h = df_h.sort_values('datetime').copy()
        out = pd.merge_asof(h, d, on='datetime', direction='backward', suffixes=('', '_dprev'))
        out['h_to_d_range_ratio'] = _safe_div((h['high'] - h['low']), out['d_range'])
        out['hour_close_to_prevday_close'] = _safe_div(h['close'], out['d_close']) - 1.0
        out['hour_vol_to_day_vol'] = _safe_div(h['volume'], out['d_vol'])
        self.columns_by_group.setdefault('cross_tf', []).extend(
            ['h_to_d_range_ratio', 'hour_close_to_prevday_close', 'hour_vol_to_day_vol']
        )
        return out[['h_to_d_range_ratio', 'hour_close_to_prevday_close', 'hour_vol_to_day_vol']]

    def build_features(self, df: pd.DataFrame, freq_hint: str = 'H', df_daily_for_cross: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        df = self._ensure_sort(df)
        feats = [
            self._base_returns(df),
            self._momentum_vol(df),
            self._volume_features(df),
            self._ta_indicators(df),
            self._tr_atr_bbands(df),
            self._calendar(df, freq_hint=freq_hint),
        ]
        if df_daily_for_cross is not None:
            feats.append(self._higher_lower_tf_links(df, df_daily_for_cross))
        X = pd.concat(feats, axis=1)
        return X, self.columns_by_group


# ---------------------------
# Targets
# ---------------------------

def target_next_log_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    close = df['close']
    return np.log(_safe_div(close.shift(-horizon), close))

def target_direction_label(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    y_reg = target_next_log_return(df, horizon=horizon)
    return (y_reg > threshold).astype(int)


# ---------------------------
# Training / Evaluation
# ---------------------------

def _xgb_model(task: str = 'regression', params: Optional[Dict] = None):
    base_params = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    if params:
        base_params.update(params)
    if task == 'regression':
        return xgb.XGBRegressor(**base_params)
    elif task == 'classification':
        base_params.setdefault('eval_metric', 'logloss')
        return xgb.XGBClassifier(**base_params)
    else:
        raise ValueError("task must be 'regression' or 'classification'")

def time_series_cv_scores(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = 'regression',
    n_splits: int = 5,
    model_params: Optional[Dict] = None,
    scale_features: bool = False,
):
    data = X.copy()
    data['__y__'] = y
    data = data.dropna().copy()
    y_clean = data.pop('__y__'); X_clean = data

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    if scale_features:
        estimator = _xgb_model(task, params=model_params)
        model_obj = Pipeline([('scaler', StandardScaler(with_mean=False)), ('xgb', estimator)])
    else:
        model_obj = _xgb_model(task, params=model_params)

    splits_used = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_clean), 1):
        X_tr, X_te = X_clean.iloc[tr_idx], X_clean.iloc[te_idx]
        y_tr, y_te = y_clean.iloc[tr_idx], y_clean.iloc[te_idx]
        model_obj.fit(X_tr, y_tr)
        if task == 'regression':
            preds = model_obj.predict(X_te)
            rmse = mean_squared_error(y_te, preds, squared=False)
            mae = mean_absolute_error(y_te, preds)
            r2 = r2_score(y_te, preds)
            fold_metrics.append({'fold': fold, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        else:
            proba = model_obj.predict_proba(X_te)[:, 1]
            preds = (proba > 0.5).astype(int)
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
            auc = roc_auc_score(y_te, proba)
            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds)
            fold_metrics.append({'fold': fold, 'AUC': auc, 'ACC': acc, 'F1': f1})
        splits_used.append((tr_idx, te_idx))

    metrics_df = pd.DataFrame(fold_metrics).set_index('fold')
    agg = metrics_df.mean().to_dict()

    # Fit final on all
    model_obj.fit(X_clean, y_clean)

    # Feature importance (gain)
    booster = model_obj.named_steps['xgb'] if isinstance(model_obj, Pipeline) else model_obj
    booster.get_booster().feature_names = list(X_clean.columns)
    importance = booster.get_booster().get_score(importance_type='gain')
    imp_df = pd.DataFrame([(k, v) for k, v in importance.items()], columns=['feature', 'gain']) \
             .sort_values('gain', ascending=False).reset_index(drop=True)

    return dict(metrics=agg, model=model_obj, importance=imp_df, used_features=list(X_clean.columns)), splits_used

def ablation_by_groups(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Dict[str, List[str]],
    task: str = 'regression',
    n_splits: int = 5,
    model_params: Optional[Dict] = None,
):
    base_res, _ = time_series_cv_scores(X, y, task=task, n_splits=n_splits, model_params=model_params)
    base_metric = next(iter(base_res['metrics'].keys()))  # first metric
    base_value = base_res['metrics'][base_metric]

    rows = [{'group': 'BASELINE', base_metric: base_value}]
    for gname, cols in groups.items():
        keep_cols = [c for c in X.columns if c not in cols]
        res, _ = time_series_cv_scores(X[keep_cols], y, task=task, n_splits=n_splits, model_params=model_params)
        rows.append({'group': f'BASE - {gname}', base_metric: res['metrics'][base_metric]})
    out = pd.DataFrame(rows)
    out['delta_vs_base'] = out[base_metric] - base_value
    return out.sort_values('group')

def compute_shap_summary(model, X_sample: pd.DataFrame, max_samples: int = 5000) -> Optional[dict]:
    if not SHAP_AVAILABLE:
        return None
    try:
        booster = model.named_steps['xgb'] if isinstance(model, Pipeline) else model
        explainer = shap.TreeExplainer(booster)
        X_small = X_sample.sample(n=min(len(X_sample), max_samples), random_state=42)
        values = explainer.shap_values(X_small)
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(-mean_abs)
        topk = min(25, X_small.shape[1])
        return {
            'top_features': [X_small.columns[i] for i in order[:topk]],
            'mean_abs_contrib': mean_abs[order[:topk]].tolist()
        }
    except Exception:
        return None


# ---------------------------
# Script main
# ---------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    # Ensure datetime dtype
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def main():
    p = argparse.ArgumentParser(description="XGBoost feature engineering & evaluation (single script)")
    p.add_argument("--hourly_csv", required=True, help="Path to HOURLY OHLCV CSV")
    p.add_argument("--daily_csv", default=None, help="Path to DAILY OHLCV CSV (optional for cross-timeframe)")
    p.add_argument("--target_kind", choices=["regression", "classification"], default="regression")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon (steps ahead)")
    p.add_argument("--n_splits", type=int, default=5, help="TimeSeriesSplit folds")
    p.add_argument("--scale_features", action="store_true", help="Apply StandardScaler before XGBoost")
    args = p.parse_args()

    df_h = load_csv(args.hourly_csv)
    df_d = load_csv(args.daily_csv) if args.daily_csv else None

    fb = FeatureBuilder()
    X, groups = fb.build_features(df_h, freq_hint='H', df_daily_for_cross=df_d)

    y = target_next_log_return(df_h, horizon=args.horizon) if args.target_kind == "regression" \
        else target_direction_label(df_h, horizon=args.horizon, threshold=0.0)

    # Train + CV
    res, splits = time_series_cv_scores(
        X, y,
        task=("classification" if args.target_kind != "regression" else "regression"),
        n_splits=args.n_splits,
        model_params=None,
        scale_features=args.scale_features
    )

    # Ablation
    ablation = ablation_by_groups(
        X, y, groups,
        task=("classification" if args.target_kind != "regression" else "regression"),
        n_splits=max(3, min(args.n_splits, 5)),
        model_params=None
    )

    # SHAP summary
    shap_summary = compute_shap_summary(res['model'], X.dropna())
    print("\n=== Aggregated CV Metrics ===")
    for k, v in res['metrics'].items():
        print(f"{k}: {v:.6f}")

    print("\n=== Top 20 Features by Gain ===")
    imp_head = res['importance'].head(20)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
        print(imp_head.to_string(index=False))

    print("\n=== Group Ablation (lower error is better; delta_vs_base > 0 means removal hurt performance) ===")
    print(ablation.to_string(index=False))

    if shap_summary:
        print("\n=== SHAP Top Features (by mean |contribution|) ===")
        for name, contrib in zip(shap_summary['top_features'], shap_summary['mean_abs_contrib']):
            print(f"{name}: {contrib:.6f}")

if __name__ == "__main__":
    main()


"""
Feature Engineering + XGBoost Evaluation Starter
================================================

This module builds a reproducible feature-engineering pipeline for HLCOV data
(High, Low, Close, Open, Volume), with:
- Rich feature groups (price, volatility, volume, indicators, calendar, cross-timeframe)
- Clean target construction (next-period return or classification label)
- TimeSeries cross-validation
- Group-wise ablation study
- XGBoost training + importance
- Optional SHAP explainability (if `shap` is installed)

Requirements (install as needed):
    pip install pandas numpy scikit-learn xgboost ta shap matplotlib

Expected input DataFrame columns:
    - "datetime" (tz-aware or naive)
    - "open", "high", "low", "close", "volume"

Author: ChatGPT
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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

# Optional: `ta` library for technical indicators (fallback to manual if not available)
try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

# Optional: SHAP explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------------------
# Utility helpers
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
    """
    Build feature groups from HLCOV data.
    """
    windows_short: Tuple[int, ...] = (3, 6, 12, 24)   # for hourly; use smaller for daily as needed
    windows_long: Tuple[int, ...]  = (48, 72, 120)    # adjust per frequency
    # Name -> list of column names produced
    columns_by_group: Dict[str, List[str]] = field(default_factory=dict)

    def _ensure_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'datetime' not in df.columns:
            raise ValueError("Input DataFrame must contain 'datetime' column.")
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def _base_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
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

        # Volume spike
        out['vol_spike_ratio_24'] = _safe_div(vol, vol.rolling(24).mean())
        # VWAP approximation within bar: (H+L+C)/3 as typical price * volume
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        out['typical_vwap'] = (tp * vol).rolling(10).sum() / vol.rolling(10).sum()

        self.columns_by_group.setdefault('volume', []).extend(out.columns.tolist())
        return out

    def _ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['volume']

        # Basic MAs
        for w in (5, 10, 20, 50, 100, 200):
            out[f'sma_{w}'] = close.rolling(w).mean()
            out[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()

        # RSI, MACD, Stoch, ADX, CCI if ta available; lightweight fallbacks otherwise
        if TA_AVAILABLE:
            out['rsi_14'] = ta.momentum.rsi(close, window=14)
            macd = ta.trend.MACD(close)
            out['macd'] = macd.macd()
            out['macd_signal'] = macd.macd_signal()
            stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
            out['stoch_k'] = stoch.stoch()
            out['stoch_d'] = stoch.stoch_signal()
            adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            out['adx_14'] = adx.adx()
            out['cci_20'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()
        else:
            delta = close.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rs = _safe_div(up, down)
            out['rsi_14'] = 100 - (100 / (1 + rs))
            # Simple MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            out['macd'] = ema12 - ema26
            out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
            # Simple stochastic
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            out['stoch_k'] = _safe_div(close - lowest_low, highest_high - lowest_low) * 100
            out['stoch_d'] = out['stoch_k'].rolling(3).mean()
            # ADX/CCI omitted when ta missing

        self.columns_by_group.setdefault('ta', []).extend(out.columns.tolist())
        return out

    def _tr_atr_bbands(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        out['true_range'] = tr
        out['atr_14'] = tr.rolling(14).mean()

        ma20 = close.rolling(20).mean()
        sd20 = close.rolling(20).std(ddof=0)
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

        # cyclical encodings
        out['dow_sin'] = _sin_time(out['dayofweek'], 7)
        out['dow_cos'] = _cos_time(out['dayofweek'], 7)
        out['dom_sin'] = _sin_time(out['dayofmonth'], 31)
        out['dom_cos'] = _cos_time(out['dayofmonth'], 31)
        out['m_sin'] = _sin_time(out['month'], 12)
        out['m_cos'] = _cos_time(out['month'], 12)

        if freq_hint.upper().startswith('H'):
            out['hour'] = dt.dt.hour
            out['hour_sin'] = _sin_time(out['hour'], 24)
            out['hour_cos'] = _cos_time(out['hour'], 24)
            out['is_weekend'] = (out['dayofweek'] >= 5).astype(int)

        self.columns_by_group.setdefault('calendar', []).extend(out.columns.tolist())
        return out

    def _higher_lower_tf_links(self, df_h: Optional[pd.DataFrame], df_d: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Cross-timeframe joins (optional). Requires hourly (h) and daily (d) DataFrames.
        Returns a frame indexed like df_h if df_h is provided, else empty.
        """
        if df_h is None or df_d is None:
            return pd.DataFrame(index=(df_h.index if df_h is not None else []))

        # Align: bring previous day's stats onto each hour of the following day
        df_d = df_d.sort_values('datetime').copy()
        df_d['d_range'] = df_d['high'] - df_d['low']
        df_d['d_vol'] = df_d['volume']
        d_cols = ['datetime', 'close', 'd_range', 'd_vol']

        # merge-asof requires numeric or datetime sorted joining keys
        h = df_h.sort_values('datetime').copy()
        d = df_d[d_cols].rename(columns={'close': 'd_close'})
        out = pd.merge_asof(
            h, d,
            on='datetime',
            direction='backward',
            suffixes=('', '_dprev')
        )
        # compute ratios
        out['h_to_d_range_ratio'] = _safe_div((h['high'] - h['low']), out['d_range'])
        out['hour_close_to_prevday_close'] = _safe_div(h['close'], out['d_close']) - 1.0
        out['hour_vol_to_day_vol'] = _safe_div(h['volume'], out['d_vol'])

        self.columns_by_group.setdefault('cross_tf', []).extend(
            ['h_to_d_range_ratio', 'hour_close_to_prevday_close', 'hour_vol_to_day_vol']
        )
        return out[['h_to_d_range_ratio', 'hour_close_to_prevday_close', 'hour_vol_to_day_vol']]

    def build_features(
        self,
        df: pd.DataFrame,
        freq_hint: str = 'H',
        df_daily_for_cross: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Return (feature_frame, columns_by_group)."""
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
            cross = self._higher_lower_tf_links(df, df_daily_for_cross)
            feats.append(cross)

        X = pd.concat(feats, axis=1)
        return X, self.columns_by_group


# ---------------------------
# Target Builders
# ---------------------------

def target_next_log_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Next-period log return over `horizon` steps."""
    close = df['close']
    return np.log(_safe_div(close.shift(-horizon), close))


def target_direction_label(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """Binary classification label: 1 if next return > threshold, else 0."""
    y_reg = target_next_log_return(df, horizon=horizon)
    return (y_reg > threshold).astype(int)


# ---------------------------
# Evaluation Utilities
# ---------------------------

@dataclass
class EvalResult:
    metrics: Dict[str, float]
    feature_importance_gain: pd.DataFrame
    model: "xgb.XGBRegressor | xgb.XGBClassifier"
    used_features: List[str]
    shap_summary: Optional[dict] = None  # key pointers only


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
) -> Tuple[EvalResult, List[Tuple[np.ndarray, np.ndarray]]]:
    """Run TimeSeriesSplit CV and fit final model on all data.
    Returns EvalResult + list of (train_idx, test_idx) used.
    """
    assert task in ('regression', 'classification')

    # Align X & y, drop rows with NaNs
    data = X.copy()
    data['__y__'] = y
    data = data.dropna().copy()

    y_clean = data.pop('__y__')
    X_clean = data

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    # optional scaling
    if scale_features:
        estimator = _xgb_model(task, params=model_params)
        pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('xgb', estimator)])
        model_obj = pipe
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
            # simple threshold 0.5
            preds = (proba > 0.5).astype(int)
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
            auc = roc_auc_score(y_te, proba)
            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds)
            fold_metrics.append({'fold': fold, 'AUC': auc, 'ACC': acc, 'F1': f1})

        splits_used.append((tr_idx, te_idx))

    # Aggregate metrics
    metrics_df = pd.DataFrame(fold_metrics).set_index('fold')
    agg = metrics_df.mean().to_dict()

    # Fit final model on all
    model_obj.fit(X_clean, y_clean)

    # Feature importance (gain)
    if isinstance(model_obj, Pipeline):
        booster = model_obj.named_steps['xgb']
    else:
        booster = model_obj

    # XGBoost get_booster() handles feature importance
    fmap = {i: col for i, col in enumerate(X_clean.columns)}
    booster.get_booster().feature_names = list(fmap.values())
    importance = booster.get_booster().get_score(importance_type='gain')
    imp_df = pd.DataFrame(
        [(k, v) for k, v in importance.items()],
        columns=['feature', 'gain']
    ).sort_values('gain', ascending=False).reset_index(drop=True)

    eval_res = EvalResult(
        metrics=agg,
        feature_importance_gain=imp_df,
        model=model_obj,
        used_features=list(X_clean.columns),
    )
    return eval_res, splits_used


def ablation_by_groups(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Dict[str, List[str]],
    task: str = 'regression',
    n_splits: int = 5,
    model_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Train baseline and then remove each group to measure delta in performance (higher delta = more valuable)."""
    base_res, _ = time_series_cv_scores(X, y, task=task, n_splits=n_splits, model_params=model_params)
    base_metric = list(base_res.metrics.items())[0][0]  # pick first metric key
    base_value = base_res.metrics[base_metric]

    rows = [{'group': 'BASELINE', base_metric: base_value}]
    for gname, cols in groups.items():
        keep_cols = [c for c in X.columns if c not in cols]
        res, _ = time_series_cv_scores(X[keep_cols], y, task=task, n_splits=n_splits, model_params=model_params)
        rows.append({'group': f'BASE - {gname}', base_metric: res.metrics[base_metric]})

    out = pd.DataFrame(rows)
    out['delta_vs_base'] = out[base_metric] - base_value
    return out.sort_values('group')


def compute_shap_summary(model, X_sample: pd.DataFrame, task: str = 'regression', max_samples: int = 5000) -> Optional[dict]:
    if not SHAP_AVAILABLE:
        return None
    try:
        if isinstance(model, Pipeline):
            booster = model.named_steps['xgb']
        else:
            booster = model
        explainer = shap.TreeExplainer(booster)
        X_small = X_sample.sample(n=min(len(X_sample), max_samples), random_state=42)
        values = explainer.shap_values(X_small)
        # Return a small, serializable summary (do not store huge arrays)
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(-mean_abs)
        topk = min(25, len(X_small.columns))
        return {
            'top_features': [X_small.columns[i] for i in order[:topk]],
            'mean_abs_contrib': mean_abs[order[:topk]].tolist()
        }
    except Exception:
        return None


# ---------------------------
# End-to-end convenience
# ---------------------------

def run_pipeline(
    df_hourly: pd.DataFrame,
    df_daily: Optional[pd.DataFrame] = None,
    target_kind: str = 'regression',
    horizon: int = 1,
    n_splits: int = 5,
    model_params: Optional[Dict] = None,
    scale_features: bool = False,
) -> Dict:
    """
    Build features (hourly, optionally cross-linked with daily), construct target,
    run CV training, compute ablation, and (optionally) SHAP.
    """
    fb = FeatureBuilder()
    X, groups = fb.build_features(df_hourly, freq_hint='H', df_daily_for_cross=df_daily)

    if target_kind == 'regression':
        y = target_next_log_return(df_hourly, horizon=horizon)
    else:
        y = target_direction_label(df_hourly, horizon=horizon, threshold=0.0)

    eval_res, splits = time_series_cv_scores(
        X, y, task=('classification' if target_kind != 'regression' else 'regression'),
        n_splits=n_splits, model_params=model_params, scale_features=scale_features
    )

    ablation = ablation_by_groups(X, y, groups, task=('classification' if target_kind!='regression' else 'regression'),
                                  n_splits=max(3, min(n_splits, 5)), model_params=model_params)

    shap_summary = compute_shap_summary(eval_res.model, X.dropna(), task=('classification' if target_kind!='regression' else 'regression'))
    eval_res.shap_summary = shap_summary

    return {
        'metrics': eval_res.metrics,
        'feature_importance_gain': eval_res.feature_importance_gain,
        'ablation': ablation,
        'shap_summary': shap_summary,
        'used_features': eval_res.used_features,
        'groups': groups,
    }


# ---------------------------
# Minimal demo with synthetic data
# ---------------------------

def _synthetic_ohlcv(n=2000, seed=42, freq='H') -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_index = pd.date_range('2023-01-01', periods=n, freq=freq)
    prices = np.cumsum(rng.normal(0, 1, size=n)).astype(float) + 100
    close = pd.Series(prices)
    open_ = close.shift(1).fillna(method='bfill')
    high = np.maximum(open_, close) + rng.random(n) * 0.5
    low  = np.minimum(open_, close) - rng.random(n) * 0.5
    volume = rng.integers(1000, 5000, size=n)
    df = pd.DataFrame({'datetime': dt_index, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    return df

if __name__ == "__main__":
    # Example usage:
    df_h = _synthetic_ohlcv(n=1500, freq='H')
    df_d = _synthetic_ohlcv(n=1500//24 + 5, freq='D')  # rough daily companion

    results = run_pipeline(
        df_hourly=df_h,
        df_daily=df_d,
        target_kind='regression',
        horizon=1,
        n_splits=4,
        model_params=None,
        scale_features=False
    )
    print("Aggregated CV Metrics:", results['metrics'])
    print("Top 10 Features by Gain:\n", results['feature_importance_gain'].head(10))
    print("Ablation (head):\n", results['ablation'].head())
    if results['shap_summary']:
        print("SHAP Top Features:", results['shap_summary']['top_features'][:10])

"""训练 LightGBM：
- 为每个分位点训练一个分位回归模型（objective='quantile', alpha=q）
- 训练风险标签分类器（risk_label）
- 训练波动率回归模型
- 保存模型并输出评估报告（coverage, MSE, AUC）

使用示例：
    python src\training\train_lightgbm.py --features data/features.parquet --outdir models
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score


DEFAULT_QUANTILES = [0.01, 0.05, 0.5, 0.95, 0.99]


def load_features(parquet_path=None):
    if parquet_path and os.path.exists(parquet_path):
        print(f"Loading features from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        # try to generate features using existing function if available
        try:
            from src.data.features import make_features
            print("Generating features via src.data.features.make_features()")
            df = make_features()
        except Exception as e:
            raise FileNotFoundError(f"Cannot load features.parquet and failed to generate features: {e}")

    return df


def prepare_tabular_data(df, feat_cols=None, dropna=True):
    # If feat_cols not provided, pick a sensible subset (columns that are numeric and not target columns)
    if feat_cols is None:
        # exclude target-like columns
        exclude = {c for c in df.columns if c.startswith('q_') or c in ['VaR95','VaR99','CVaR95','volatility','trend','risk_label','ret']}
        feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feat_cols].copy()
    # simple fill first (forward/backward), then convert to numpy and sanitize
    X = X.ffill().bfill().fillna(0)

    # convert to numpy float array and ensure no inf/NaN/huge values
    X_arr = X.values.astype(float)
    total_nonfinite = int(np.count_nonzero(~np.isfinite(X_arr)))
    if total_nonfinite > 0:
        print(f"⚠️ Found {total_nonfinite} non-finite entries in tabular features; replacing with column medians")
        tmp = X_arr.copy()
        tmp[~np.isfinite(tmp)] = np.nan
        # compute column medians ignoring nan
        col_median = np.nanmedian(tmp, axis=0)
        # where median is nan (all values invalid), replace with 0.0
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        # broadcast replace
        inds = np.where(~np.isfinite(X_arr))
        if inds[0].size > 0:
            X_arr[inds] = col_median[inds[1]]
        remaining_nonfinite = int(np.count_nonzero(~np.isfinite(X_arr)))
        if remaining_nonfinite > 0:
            print(f"⚠️ After replacement, still {remaining_nonfinite} non-finite entries remain; they will be set to 0")
            X_arr[~np.isfinite(X_arr)] = 0.0
    # clip extreme values to avoid overflow issues in sklearn/lightgbm
    X_arr = np.clip(X_arr, -1e6, 1e6)

    y_quantiles = {q: df.get(f"q_{int(q*100):02d}").values if f"q_{int(q*100):02d}" in df.columns else None for q in DEFAULT_QUANTILES}
    y_vol = df.get('volatility').values if 'volatility' in df.columns else None
    y_risk = df.get('risk_label').values if 'risk_label' in df.columns else None

    return X_arr, feat_cols, y_quantiles, y_vol, y_risk


def train_quantile_models(X, y_true, quantiles=DEFAULT_QUANTILES, feature_names=None, outdir='models'):
    models = {}
    evals = {}
    for q in quantiles:
        col_name = f"q_{int(q*100):02d}"
        if y_true.get(q) is None:
            print(f"Skipping quantile {q}: target not available")
            continue

        y = y_true[q]
        # remove NaNs
        mask = ~pd.isna(y)
        X_sub = X[mask]
        y_sub = y[mask]

        if len(y_sub) < 10:
            print(f"Not enough samples for quantile {q}, skipping")
            continue

        X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        print(f"Training LightGBM quantile model q={q} (samples: {len(y_sub)})")
        # some lightgbm versions don't accept verbose in fit(); omit to keep compatibility
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        except TypeError:
            # fallback: call without eval_set
            model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        # coverage: for lower quantiles (q<0.5) we expect y <= pred fraction ~ q; for upper (q>0.5) expect y >= pred fraction ~ q
        if q < 0.5:
            coverage = float((y_val <= pred_val).mean())
        else:
            coverage = float((y_val >= pred_val).mean())

        models[q] = model
        evals[q] = {'coverage': coverage, 'val_mse': float(mean_squared_error(y_val, pred_val)), 'n_val': len(y_val)}

        # save model
        Path(outdir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(outdir, f'lgb_quantile_q{int(q*100):02d}.pkl')
        joblib.dump(model, model_path)
        print(f"Saved quantile model: {model_path}")

    return models, evals


def train_risk_classifier(X, y_risk, outdir='models'):
    result = None
    if y_risk is None:
        print("No risk_label available, skipping risk classifier training")
        return None, None

    mask = ~pd.isna(y_risk)
    X_sub = X[mask]
    y_sub = y_risk[mask].astype(int)

    if len(np.unique(y_sub)) < 2 or len(y_sub) < 30:
        print("Not enough samples or classes for risk classifier, skipping")
        return None, None

    X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    print(f"Training risk classifier (samples: {len(y_sub)})")
    clf.fit(X_train, y_train)

    pred_val_proba = clf.predict_proba(X_val)
    try:
        # if binary, take probability for class 1
        if pred_val_proba.shape[1] == 2:
            auc = float(roc_auc_score(y_val, pred_val_proba[:, 1]))
        else:
            # multiclass: macro average using one-vs-rest
            auc = float(roc_auc_score(y_val, pred_val_proba, multi_class='ovr'))
    except Exception:
        auc = None

    acc = float(accuracy_score(y_val, clf.predict(X_val)))

    Path(outdir).mkdir(parents=True, exist_ok=True)
    clf_path = os.path.join(outdir, 'lgb_risk_classifier.pkl')
    joblib.dump(clf, clf_path)
    print(f"Saved risk classifier: {clf_path}")

    evals = {'auc': auc, 'accuracy': acc, 'n_val': len(y_val)}
    return clf, evals


def train_volatility_regressor(X, y_vol, outdir='models'):
    if y_vol is None:
        print("No volatility target available, skipping vol regressor training")
        return None, None

    mask = ~pd.isna(y_vol)
    X_sub = X[mask]
    y_sub = y_vol[mask]

    if len(y_sub) < 30:
        print("Not enough samples for volatility regressor, skipping")
        return None, None

    X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

    reg = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    print(f"Training volatility regressor (samples: {len(y_sub)})")
    reg.fit(X_train, y_train)

    pred_val = reg.predict(X_val)
    mse = float(mean_squared_error(y_val, pred_val))

    Path(outdir).mkdir(parents=True, exist_ok=True)
    reg_path = os.path.join(outdir, 'lgb_volatility_regressor.pkl')
    joblib.dump(reg, reg_path)
    print(f"Saved volatility regressor: {reg_path}")

    evals = {'mse': mse, 'n_val': len(y_val)}
    return reg, evals


def save_eval_summary(summary: dict, outdir='models'):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(outdir, 'lgb_eval_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation summary: {path}")


def main(args):
    df = load_features(args.features)

    X, feature_names, y_quantiles, y_vol, y_risk = prepare_tabular_data(df)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.outdir, 'lgb_scaler.pkl'))

    summary = {'quantiles': {}, 'risk': None, 'volatility': None}

    models_q, evals_q = train_quantile_models(X_scaled, y_quantiles, quantiles=args.quantiles, feature_names=feature_names, outdir=args.outdir)
    summary['quantiles'] = evals_q

    clf, eval_risk = train_risk_classifier(X_scaled, y_risk, outdir=args.outdir)
    summary['risk'] = eval_risk

    vol_model, eval_vol = train_volatility_regressor(X_scaled, y_vol, outdir=args.outdir)
    summary['volatility'] = eval_vol

    save_eval_summary(summary, outdir=args.outdir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='data/features.parquet', help='path to features parquet')
    p.add_argument('--outdir', default='models', help='directory to save models')
    p.add_argument('--quantiles', nargs='+', type=float, default=DEFAULT_QUANTILES, help='list of quantiles')
    args = p.parse_args()
    main(args)

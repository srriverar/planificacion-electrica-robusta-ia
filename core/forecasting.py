# -*- coding: utf-8 -*-
"""
Módulo de minería de datos y predicción con inteligencia computacional.

- Aprendizaje no supervisado: K-Means sobre perfiles diarios y PCA de ciudades (OE-1).
- Predicción a corto plazo (1–24 h): persistencia, seasonal-naive, MLP (red neuronal),
  Gradient Boosting y ensemble apilado con meta-learner Ridge (OE-2/OE-3).
- Incertidumbre: bandas de cuantiles (Gradient Boosting quantile) y desviación de residuos.
- Interpretabilidad: importancia por permutación y dependencia parcial (actividad 6 de la propuesta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from . import synthetic_data as sd

HORIZONS = (1, 6, 12, 24)


# ---------------------------------------------------------------------------
# OE-1 · minería de datos no supervisada
# ---------------------------------------------------------------------------
def cluster_daily_profiles(city: str, k: int = 4, years: int = 2) -> dict:
    """K-Means sobre perfiles diarios normalizados → patrones de consumo."""
    profiles = sd.daily_profiles(city, years=years)
    X = profiles.to_numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    centers = pd.DataFrame(km.cluster_centers_, columns=profiles.columns)
    inertia = []
    for kk in range(2, 9):
        inertia.append(KMeans(n_clusters=kk, n_init=5, random_state=42).fit(X).inertia_)
    names = {}
    for c in range(k):
        prof = centers.loc[c]
        pico_h = int(prof.values.argmax())
        valle_h = int(prof.values.argmin())
        names[c] = f"Clúster {c} · pico {pico_h:02d}h · valle {valle_h:02d}h · n={int((labels==c).sum())}"
    return dict(profiles=profiles, labels=labels, centers=centers, names=names,
                inertia=pd.Series(inertia, index=range(2, 9)), km=km)


def pca_cities(n_comp: int = 2) -> dict:
    """PCA sobre la matriz de indicadores urbanos (16 ciudades)."""
    feats = sd.city_feature_matrix()
    X = feats.drop(columns=["region"]).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=42)
    Y = pca.fit_transform(Xs)
    loadings = pd.DataFrame(pca.components_.T,
                            index=feats.drop(columns=["region"]).columns,
                            columns=[f"PC{i+1}" for i in range(n_comp)])
    out = pd.DataFrame(Y, index=feats.index, columns=[f"PC{i+1}" for i in range(n_comp)])
    out["region"] = feats["region"]
    return dict(scores=out, loadings=loadings,
                explained=pca.explained_variance_ratio_,
                features=feats.drop(columns=["region"]))


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Reporte sintético de calidad de datos (minería): NaN, outliers, rango."""
    rows = []
    for col in df.columns:
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
        rows.append(dict(variable=col, n=len(s), nan_pct=round(100 * s.isna().mean(), 2),
                         outliers_iqr3=outliers, min=round(float(s.min()), 2),
                         max=round(float(s.max()), 2), mean=round(float(s.mean()), 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OE-2/OE-3 · predicción a corto plazo
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Matriz de características para predicción directa a `horizon` horas."""
    f = pd.DataFrame(index=df.index)
    idx = df.index
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["temp_c"] = df["temp_c"]
    f["solar_pu"] = df["solar_pu"]
    f["wind_pu"] = df["wind_pu"]
    f["ci_gco2"] = df["ci_gco2_kwh"]
    f["price"] = df["price_eur_mwh"]
    f["dc_mw"] = df["dc_mw"]
    f["ev_mw"] = df["ev_mw"]
    d = df["demand_mw"]
    f["lag_24"] = d.shift(24)
    f["lag_48"] = d.shift(48)
    f["lag_168"] = d.shift(168)
    f["roll_24_mean"] = d.shift(1).rolling(24).mean()
    f["roll_168_mean"] = d.shift(1).rolling(168).mean()
    f["target"] = d.shift(-horizon)
    return f.dropna()


def _metrics(y_true, y_pred) -> dict:
    return dict(
        MAPE_pct=float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        RMSE_MW=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        R2=float(r2_score(y_true, y_pred)),
    )


def train_forecast(city: str, horizon: int = 24, years: int = 2) -> dict:
    """Entrena baselines + MLP + GBR + ensemble apilado y evalúa en test."""
    df = sd.city_hourly(city, years=years)
    f = build_features(df, horizon)
    feat_cols = [c for c in f.columns if c != "target"]
    X, y = f[feat_cols].to_numpy(), f["target"].to_numpy()
    n = len(X)
    n_tr, n_va = int(n * 0.70), int(n * 0.85)
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:n_va], y[n_tr:n_va]
    Xte, yte = X[n_va:], y[n_va:]
    test_idx = f.index[n_va:]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    # Baselines
    y_persist = f["lag_24"].to_numpy()[n_va:]                       # persistencia (t+24 ≈ t)
    y_seasonal = f["lag_168"].to_numpy()[n_va:]                      # naive estacional semanal
    # ajustar el offset correcto: target(t)=demand(t+h); lag_24 está alineado a t
    base = {"Persistencia (24 h)": _metrics(yte, y_persist),
            "Naive estacional (168 h)": _metrics(yte, y_seasonal)}

    # Red neuronal (MLP)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                       alpha=1e-4, max_iter=250, early_stopping=True,
                       validation_fraction=0.12, random_state=42, learning_rate_init=3e-3)
    mlp.fit(Xtr_s, ytr)
    p_mlp_te = mlp.predict(Xte_s)
    p_mlp_va = mlp.predict(Xva_s)

    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.06,
                                    subsample=0.9, random_state=42)
    gbr.fit(Xtr, ytr)
    p_gbr_te = gbr.predict(Xte)
    p_gbr_va = gbr.predict(Xva)

    models = {"Red neuronal (MLP 64-32)": p_mlp_te, "Gradient Boosting": p_gbr_te}
    scores = dict(base)
    scores["Red neuronal (MLP 64-32)"] = _metrics(yte, p_mlp_te)
    scores["Gradient Boosting"] = _metrics(yte, p_gbr_te)

    # Ensemble apilado (Ridge sobre predicciones de validación)
    stack_va = np.column_stack([p_mlp_va, p_gbr_va])
    stack_te = np.column_stack([p_mlp_te, p_gbr_te])
    meta = Ridge(alpha=1.0, positive=True).fit(stack_va, yva)
    p_ens_te = meta.predict(stack_te)
    scores["Ensemble (stacking Ridge)"] = _metrics(yte, p_ens_te)

    # Bandas de incertidumbre: cuantiles GBR
    q05 = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=150,
                                    max_depth=4, learning_rate=0.07, random_state=42).fit(Xtr, ytr)
    q95 = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=150,
                                    max_depth=4, learning_rate=0.07, random_state=42).fit(Xtr, ytr)
    lo = q05.predict(Xte)
    hi = q95.predict(Xte)
    resid_std = float(np.std(yte - p_ens_te))

    # Comparación con metas de la propuesta
    best_single = min(scores["Red neuronal (MLP 64-32)"]["MAPE_pct"],
                      scores["Gradient Boosting"]["MAPE_pct"])
    ens_mape = scores["Ensemble (stacking Ridge)"]["MAPE_pct"]
    mejora_ens_pct = (best_single - ens_mape) / best_single * 100 if best_single > 0 else 0.0

    test_df = pd.DataFrame({
        "real": yte, "mlp": p_mlp_te, "gbr": p_gbr_te, "ensemble": p_ens_te,
        "q05": lo, "q95": hi, "persistencia": y_persist,
    }, index=test_idx)

    # Importancia por permutación sobre el GBR (interpretabilidad)
    perm = permutation_importance(gbr, Xte, yte, n_repeats=6, random_state=42,
                                  scoring="neg_mean_absolute_percentage_error")
    imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)

    return dict(scores=pd.DataFrame(scores).T.sort_values("MAPE_pct"),
                test_df=test_df, resid_std=resid_std,
                mejora_ensemble_pct=mejora_ens_pct,
                best_single_mape=best_single, ensemble_mape=ens_mape,
                importance=imp, feat_cols=feat_cols,
                gbr=gbr, Xte=Xte, yte=yte, meta_weights=meta.coef_,
                n_train=n_tr, n_val=n_va - n_tr, n_test=n - n_va)


def partial_dep_temp(gbr, X, feat_cols, feature="temp_c"):
    """Dependencia parcial de la demanda predicha respecto a una variable."""
    j = feat_cols.index(feature)
    pd_res = partial_dependence(gbr, X, [j], grid_resolution=12, kind="average")
    grid = pd_res.get("grid_values", pd_res.get("values"))  # sklearn >=1.0 usa grid_values
    return pd.Series(pd_res["average"][0], index=grid[0])


def learning_curve_gbr(Xtr, ytr, Xte, yte, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Curva de aprendizaje rápida con GBR reducido."""
    out = []
    for fr in fractions:
        m = int(len(Xtr) * fr)
        g = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08,
                                      random_state=42).fit(Xtr[:m], ytr[:m])
        out.append(dict(fraction=fr, n=m, **_metrics(yte, g.predict(Xte))))
    return pd.DataFrame(out)

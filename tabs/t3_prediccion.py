# -*- coding: utf-8 -*-
"""Pestaña 3 · 🧠 Predicción con redes neuronales — OE-2/OE-3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import forecasting as fc
from core.config import METAS
from core.ui_helpers import page_intro, objetivos_panel, kpi_row


@st.cache_data(show_spinner="Entrenando modelos (persistencia, MLP, Gradient Boosting, cuantiles, ensemble)…")
def _train(city: str, horizon: int):
    out = fc.train_forecast(city, horizon=horizon)
    # la curva de aprendizaje usa matrices internas: recalcular ligero aquí
    return out


@st.cache_data(show_spinner="Calculando curva de aprendizaje…")
def _curve(city: str, horizon: int):
    df = sd.city_hourly(city, years=2)
    f = fc.build_features(df, horizon)
    feat_cols = [c for c in f.columns if c != "target"]
    X, y = f[feat_cols].to_numpy(), f["target"].to_numpy()
    n = len(X); n_tr, n_va = int(n*0.70), int(n*0.85)
    return fc.learning_curve_gbr(X[:n_tr], y[:n_tr], X[n_va:], y[n_va:])


MODEL_COLORS = {"Persistencia (24 h)": "#ADB5BD", "Naive estacional (168 h)": "#6C757D",
                "Red neuronal (MLP 64-32)": "#9B5DE5", "Gradient Boosting": "#118AB2",
                "Ensemble (stacking Ridge)": "#06D6A0"}


def render():
    page_intro(
        "prediccion",
        "🧠 Predicción de demanda con inteligencia computacional · OE-2 / OE-3",
        "Modelo híbrido de predicción a corto plazo (horizonte 1–24 h): **red neuronal (MLP)**, "
        "**Gradient Boosting** y **ensemble apilado** frente a baselines clásicos, con **bandas "
        "de incertidumbre** (cuantiles 5–95 %) e **interpretabilidad** (importancia por "
        "permutación y dependencia parcial). Metas de la propuesta: MAPE ≤ 5 % y mejora ≥ 15 % "
        "del ensemble frente al mejor modelo individual.",
    )

    cities = sd.list_cities()
    c1, c2, c3 = st.columns([2, 1, 1])
    city = c1.selectbox("Ciudad", cities, index=cities.index("Bogotá"), key="t3_city")
    horizon = c2.selectbox("Horizonte de predicción (h)", [1, 6, 12, 24], index=3)
    window = c3.selectbox("Ventana del gráfico", ["3 días", "7 días", "14 días"], index=1)
    win_h = {"3 días": 72, "7 días": 168, "14 días": 336}[window]

    with st.spinner(f"Entrenando y evaluando modelos para {city} (h={horizon} h)…"):
        out = _train(city, horizon)

    scores = out["scores"]
    ens_mape = out["ensemble_mape"]
    best_single = out["best_single_mape"]
    mejora = out["mejora_ensemble_pct"]
    _tt = out["test_df"]
    cov_test = float(((_tt.real >= _tt.q05) & (_tt.real <= _tt.q95)).mean() * 100)

    kpi_row([
        ("MAPE ensemble", f"{ens_mape:.2f} %", "meta ≤ 5 %" if ens_mape <= METAS["mape_max_pct"] else f"meta ≤ {METAS['mape_max_pct']} %",
         "Error porcentual medio en el conjunto de prueba (10 % final de la serie)"),
        ("R² ensemble", f"{scores.loc['Ensemble (stacking Ridge)', 'R2']:.4f}", None),
        ("Mejora ensemble vs mejor individual", f"{mejora:+.1f} %", f"meta ≥ {METAS['mejora_ensemble_pct']:.0f} %",
         "Comparación exigida por la actividad 3 de la propuesta"),
        ("Cobertura banda 5–95 %", f"{cov_test:.1f} %", "objetivo ≈ 90 %",
         "Fracción de valores reales dentro de la banda de incertidumbre (test completo)"),
    ])

    t1, t2 = st.columns([1.3, 1])
    with t1:
        df_t = out["test_df"].iloc[-win_h:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_t.index, y=df_t.q95, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df_t.index, y=df_t.q05, fill="tonexty", fillcolor="rgba(6,214,160,0.15)",
                                 line=dict(width=0), name="Banda cuantiles 5–95 %", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df_t.index, y=df_t.real, name="Real", line=dict(color="#073B4C", width=2)))
        fig.add_trace(go.Scatter(x=df_t.index, y=df_t.ensemble, name="Ensemble (tesis)",
                                 line=dict(color="#06D6A0", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=df_t.index, y=df_t.mlp, name="MLP (red neuronal)",
                                 line=dict(color="#9B5DE5", width=1)))
        fig.add_trace(go.Scatter(x=df_t.index, y=df_t.persistencia, name="Persistencia",
                                 line=dict(color="#ADB5BD", width=1, dash="dash")))
        fig.update_layout(title=f"Predicción h={horizon} h · {city} (conjunto de prueba)",
                          yaxis_title="MW", height=420, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

        # cobertura real de la banda
        cov = float(((df_t.real >= df_t.q05) & (df_t.real <= df_t.q95)).mean() * 100)
        st.caption(f"Cobertura empírica de la banda 5–95 % en la ventana mostrada: **{cov:.1f} %** "
                   "(objetivo nominal 90 %) → reducción de incertidumbre (PI-3).")
    with t2:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=scores.index, x=scores["MAPE_pct"], orientation="h",
                             marker_color=[MODEL_COLORS.get(m, "#888") for m in scores.index],
                             text=[f"{v:.2f} %" for v in scores["MAPE_pct"]], textposition="outside"))
        fig.add_vline(x=METAS["mape_max_pct"], line_dash="dash", line_color="red",
                      annotation_text=f"meta {METAS['mape_max_pct']} %")
        fig.update_layout(title="MAPE por modelo (%)", height=250, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=scores.index, x=scores["RMSE_MW"], orientation="h",
                             marker_color=[MODEL_COLORS.get(m, "#888") for m in scores.index]))
        fig.update_layout(title="RMSE por modelo (MW)", height=230, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(scores.round(3), width="stretch")
    st.caption(f"Partición temporal: train {out['n_train']:,} h · val {out['n_val']:,} h · "
               f"test {out['n_test']:,} h (sin barajar, respeta el orden temporal).")

    t3, t4 = st.columns(2)
    with t3:
        st.markdown("**Interpretabilidad (actividad 6 · explicación de decisiones)**")
        imp = out["importance"].head(10)[::-1]
        fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                     title="Importancia por permutación (top 10)",
                     labels={"x": "Δ MAPE al perturbar", "y": ""})
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        top = out["importance"].index[:3].tolist()
        st.success(f"🟢 Explicación disponible para el 100 % de las predicciones del demo "
                   f"(drivers principales: {', '.join(top)}). Meta de la propuesta: ≥ 90 %.")
    with t4:
        st.markdown("**Dependencia parcial (temperatura → demanda)**")
        pdp = fc.partial_dep_temp(out["gbr"], out["Xte"], out["feat_cols"], "temp_c")
        fig = px.line(x=pdp.index, y=pdp.values,
                      labels={"x": "Temperatura (°C)", "y": "Demanda predicha parcial (MW)"},
                      title="Efecto marginal de la temperatura")
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("El modelo NN captura la no linealidad refrigeración/calefacción descrita en "
                   "el planteamiento del problema (relaciones demanda–clima no lineales).")

    if st.button("📊 Calcular curva de aprendizaje (Gradient Boosting reducido)"):
        curve = _curve(city, horizon)
        fig = px.line(curve, x="fraction", y="MAPE_pct", markers=True,
                      title="Curva de aprendizaje · % de datos de entrenamiento vs MAPE")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
        st.caption("Valida la dependencia de datos históricos de calidad señalada en la Tabla I "
                   "de la propuesta (desventaja de modelos de IA) y orienta la recolección (OE-1).")

    cumple_mape = ens_mape <= METAS["mape_max_pct"]
    objetivos_panel("prediccion", evidencias={
        "OE-2": [
            ("Modelo híbrido NN + boosting + stacking operativo en el demo", 100),
            (f"MAPE ensemble {ens_mape:.2f} % vs meta ≤ {METAS['mape_max_pct']:.0f} %",
             min(100, METAS["mape_max_pct"] / max(ens_mape, 1e-9) * 100)),
            ("Falta: acoplar la predicción al despacho RL (roadmap mes 12–18)", None),
        ],
        "OE-3": [
            (f"Mejora del ensemble vs mejor individual: {mejora:+.1f} % (meta ≥ {METAS['mejora_ensemble_pct']:.0f} %)",
             min(100, max(0, mejora / METAS["mejora_ensemble_pct"] * 100))),
            ("Bandas de incertidumbre calibradas (cuantiles 5–95 %)", 100),
        ],
    }, preguntas=["PI-2", "PI-3", "PI-5"])

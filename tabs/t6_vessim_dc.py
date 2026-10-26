# -*- coding: utf-8 -*-
"""Pestaña 6 · 🖥️ Data centers con Vessim — OE-4 / PI-1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import vessim_dc as vdc
from core.ui_helpers import page_intro, objetivos_panel, kpi_row

POL_COLORS = {"Sin gestión (DC tradicional)": "#ADB5BD",
              "Batería + despacho por defecto": "#FFB703",
              "Carbon-aware (tesis)": "#06D6A0"}


@st.cache_data(show_spinner="Co-simulando microred del data center con Vessim (3 políticas)…")
def _compare(city, days, step_min, pue, solar_frac, battery_frac, workload_shift):
    df = vdc.compare_policies(city, days=days, step_min=step_min, pue=pue,
                              solar_frac=solar_frac, battery_frac=battery_frac,
                              workload_shift=workload_shift)
    return df


@st.cache_data(show_spinner="Co-simulando detalle (Vessim)…")
def _detail(city, days, step_min, pue, solar_frac, battery_frac, workload_shift, policy):
    r = vdc.run_dc_microgrid(city, days=days, step_min=step_min, pue=pue,
                             solar_frac=solar_frac, battery_frac=battery_frac,
                             policy=policy, workload_shift=workload_shift)
    df = r["df"].reset_index()
    return r["kpis"], df, r["config"]


def render():
    page_intro(
        "vessim",
        "🖥️ Hiperdatacenters urbanos: co-simulación con Vessim · OE-4 / PI-1",
        "**Vessim** (co-simulador de microredes para sistemas energy-aware) modela la interacción "
        "entre la carga computacional de un data center (inferencia diurna + entrenamiento de IA "
        "nocturno × PUE), PV on-site, baterías y la red pública de la ciudad, con señales horarias "
        "de **intensidad de carbono** y **precio**. Se comparan tres políticas de gestión, incluida "
        "una **carbon-aware** propia. Esto cuantifica el impacto de la nueva tecnología que motivó "
        "la alerta de ProBogotá sobre 'hiperdatacenters' (PI-1).",
    )

    cities = sd.list_cities()
    c1, c2, c3 = st.columns([2, 1, 1])
    city = c1.selectbox("Ciudad", cities, index=cities.index("Bogotá"), key="t6_city")
    days = c2.selectbox("Horizonte", [7, 14, 30], index=2, format_func=lambda d: f"{d} días", key="t6_days")
    step_min = c3.selectbox("Paso de simulación", [15, 30], index=0, format_func=lambda s: f"{s} min", key="t6_step")

    c4, c5, c6, c7 = st.columns(4)
    pue = c4.slider("PUE", 1.10, 1.70, 1.30, 0.05, help="Power Usage Effectiveness: 1.0 ideal; 1.3 bueno; 1.6 antiguo")
    solar_frac = c5.slider("PV on-site (× pipeline MW)", 0.0, 1.5, 0.50, 0.05)
    battery_frac = c6.slider("Batería (MWh × pipeline MW)", 0.0, 3.0, 1.00, 0.1)
    workload_shift = c7.checkbox("Shift de workloads (carbon-aware)", value=True,
                                 help="Mueve ~18 % del entrenamiento flexible a horas de baja intensidad de carbono")

    c = sd.get_city(city)
    st.info(f"**{city}** — pipeline de data centers: **{c['dc_pipeline_mw']} MW** "
            f"({vdc._city_avg_mw(city):,.0f} MW de demanda media urbana ⇒ "
            f"el DC en fase activa ≈ {100*c['dc_pipeline_mw']*0.6/max(vdc._city_avg_mw(city),1):.1f} % de la ciudad).")

    cmpdf = _compare(city, days, step_min, pue, solar_frac, battery_frac, workload_shift)
    base = cmpdf.iloc[0]
    best = cmpdf[cmpdf.politica == "Carbon-aware (tesis)"].iloc[0]
    d_emi = (best.emisiones_tco2 - base.emisiones_tco2) / max(base.emisiones_tco2, 1e-9) * 100
    d_cost = (best.costo_red_eur - base.costo_red_eur) / max(base.costo_red_eur, 1e-9) * 100
    kpi_row([
        ("Carga DC (periodo)", f"{best.carga_dc_mwh:,.0f} MWh", None, f"{days} días, PUE {pue}"),
        ("Autosuficiencia (carbon-aware)", f"{best.autosuficiencia_pct:.1f} %", None,
         "Fracción de la carga cubierta sin importar de la red"),
        ("Δ Emisiones vs sin-gestión", f"{d_emi:+.1f} %", None, "Despacho carbon-aware + shift de workloads"),
        ("Δ Costo red vs sin-gestión", f"{d_cost:+.1f} %", None),
    ])

    tA, tB = st.columns(2)
    with tA:
        fig = go.Figure()
        for m, name in [("emisiones_tco2", "Emisiones (tCO₂)"), ("costo_red_eur", "Costo red (€)")]:
            fig.add_trace(go.Bar(x=cmpdf["politica"], y=cmpdf[m], name=name))
        fig.update_layout(barmode="group", title=f"Impacto por política · {days} días", height=360,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with tB:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cmpdf["politica"], y=cmpdf["importado_red_mwh"],
                             name="Importado red (MWh)", marker_color="#118AB2"))
        fig.add_trace(go.Bar(x=cmpdf["politica"], y=cmpdf["pv_generado_mwh"],
                             name="PV on-site (MWh)", marker_color="#FFB703"))
        fig.add_trace(go.Scatter(x=cmpdf["politica"], y=cmpdf["pico_import_mw"],
                                 name="Pico import (MW)", mode="lines+markers",
                                 line=dict(color="#EF476F", width=2)))
        fig.update_layout(title="Energía y potencia por política", height=360,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(cmpdf[["politica", "carga_dc_mwh", "pv_generado_mwh", "importado_red_mwh",
                        "autosuficiencia_pct", "emisiones_tco2", "costo_red_eur",
                        "pico_import_mw", "share_demanda_ciudad_pct"]].round(2),
                 width="stretch", hide_index=True)

    st.markdown("#### 🔬 Detalle horario (Vessim · Recorder)")
    policy = st.selectbox("Política a graficar", ["carbon_aware", "bateria_default", "sin_gestion"],
                          format_func=lambda p: vdc.POLITICA_LABEL[p], key="t6_policy")
    kpis, df, conf = _detail(city, days, step_min, pue, solar_frac, battery_frac,
                             workload_shift and policy == "carbon_aware", policy)
    time_col = "time" if "time" in df.columns else df.columns[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df["actor_states.it_cooling.power"] / 1e6,
                             name="Carga DC (MW)", line=dict(color="#073B4C", width=2)))
    fig.add_trace(go.Scatter(x=df[time_col], y=df["actor_states.pv_onsite.power"] / 1e6,
                             name="PV on-site (MW)", line=dict(color="#FFB703", width=2)))
    fig.add_trace(go.Scatter(x=df[time_col], y=-df["p_grid"] / 1e6,
                             name="Import red (MW)", fill="tozeroy",
                             line=dict(color="#EF476F", width=1)))
    fig.update_layout(title=f"Potencias horarias · {vdc.POLITICA_LABEL[policy]} · {city}",
                      yaxis_title="MW", height=380, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    soc_col = [cc for cc in df.columns if "soc" in cc]
    ci_col = [cc for cc in df.columns if "carbon_intensity" in cc]
    if soc_col and ci_col:
        cA, cB = st.columns(2)
        with cA:
            fig = px.area(x=df[time_col], y=df[soc_col[0]],
                          title="Estado de carga de la batería (SOC)")
            fig.update_layout(height=280, yaxis_title="SOC (0–1)")
            st.plotly_chart(fig, width="stretch")
        with cB:
            fig = px.line(x=df[time_col], y=df[ci_col[0]],
                          title="Intensidad de carbono de la red (gCO₂/kWh)")
            fig.update_layout(height=280, yaxis_title="gCO₂/kWh")
            st.plotly_chart(fig, width="stretch")
        st.caption("La política carbon-aware carga la batería cuando la intensidad de carbono es "
                   "baja y descarga cuando es alta: la SOC sigue la señal de carbono (despacho "
                   "consciente de carbono, OE-4).")

    objetivos_panel("vessim", evidencias={
        "OE-4": [
            ("Co-simulación data center ↔ red ↔ PV ↔ batería operativa con Vessim", 100),
            (f"Reducción de emisiones por gestión carbon-aware: {d_emi:+.1f} %", min(100, max(0, -d_emi) * 5)),
            ("Impacto del DC sobre la demanda urbana cuantificado (share %)", 100),
        ],
    }, preguntas=["PI-1", "PI-5"])

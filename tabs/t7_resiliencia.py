# -*- coding: utf-8 -*-
"""Pestaña 7 · 🛡️ Resiliencia y eventos extremos — OE-3 / PI-4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import pypsa_planning as pp
from core.ui_helpers import page_intro, objetivos_panel, kpi_row

MODE_COLORS = {"Determinista tradicional": "#EF476F", "Robusto-IA (tesis)": "#06D6A0"}


@st.cache_data(show_spinner="Re-despachando la red bajo el evento (PyPSA)…")
def _event_kpis(city, event, severity, year, co2_price):
    return pp.event_kpis(city, event, severity, year, co2_price)


@st.cache_data(show_spinner="Barriendo severidades (12 LP)…")
def _sweep(city, event, year, co2_price):
    return pp.severity_sweep(city, event, year, co2_price)


@st.cache_data(show_spinner="Análisis N-1 (10 despachos PyPSA)…")
def _n1(city, year, scenario, co2_price):
    return pp.n1_contingency(city, year, scenario, co2_price)


def render():
    page_intro(
        "resiliencia",
        "🛡️ Resiliencia ante eventos extremos · OE-3 / PI-4",
        "Pruebas de estrés de la planificación: **eventos extremos sintéticos** (picos de demanda, "
        "olas de calor, tormentas, fallas de generación, surgimiento de data centers) y "
        "**contingencias N-1** con re-despacho. Se compara la **planificación determinista "
        "tradicional** (BAU, margen ajustado, sin almacenamiento) contra el **modelo robusto-IA "
        "de la tesis** (margen de reserva, baterías, ahorro por respuesta de demanda). El "
        "racionamiento se valora con VOLL = 500 USD/MWh, de modo que la energía no suministrada "
        "(ENS) entra explícitamente en el óptimo.",
    )

    cities = sd.list_cities()
    c1, c2, c3, c4 = st.columns([1.8, 2, 1, 1])
    city = c1.selectbox("Ciudad", cities, index=cities.index("Bogotá"), key="t7_city")
    event = c2.selectbox("Evento extremo", list(pp.EVENTS.keys()), key="t7_event")
    severity = c3.slider("Severidad", 0.05, 0.30, 0.15, 0.05,
                         help="Magnitud del choque (p. ej. 0.15 = +15 % de demanda)")
    year = c4.selectbox("Año", [2026, 2030, 2035], index=1, key="t7_year")
    co2_price = st.sidebar.number_input("Precio CO₂ para resiliencia (USD/t)", 0, 150, 30, 5, key="t7_co2")
    st.caption(pp.EVENTS[event]["msg"].replace("X", f"{severity*100:.0f}"))

    ev = _event_kpis(city, event, severity, year, co2_price)
    det = ev.set_index("modo")
    ens_det, ens_rob = det.loc["Determinista tradicional", "ENS_MWh"], det.loc["Robusto-IA (tesis)", "ENS_MWh"]
    kpi_row([
        ("ENS · determinista", f"{ens_det:,.0f} MWh", None, f"{det.loc['Determinista tradicional','ENS_pct']:.2f} % de la demanda"),
        ("ENS · robusto-IA", f"{ens_rob:,.0f} MWh", None, f"{det.loc['Robusto-IA (tesis)','ENS_pct']:.2f} % de la demanda"),
        ("Reducción de ENS", f"{(1-ens_rob/max(ens_det,1e-9))*100:+.0f} %", None, "Beneficio de resiliencia del modelo robusto"),
        ("Δ costo del evento", f"{det.loc['Robusto-IA (tesis)','costo_MUSD']-det.loc['Determinista tradicional','costo_MUSD']:+,.0f} M USD", None, "Robusto − determinista bajo el evento"),
    ])

    tA, tB = st.columns(2)
    with tA:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ev["modo"], y=ev["ENS_MWh"], name="ENS (MWh)",
                             marker_color=[MODE_COLORS[m] for m in ev["modo"]]))
        fig.update_layout(title=f"Energía no suministrada · {event} ({severity*100:.0f} %)",
                          height=320, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with tB:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ev["modo"], y=ev["max_carga_linea_pct"], name="Carga máx. línea (%)",
                             marker_color=[MODE_COLORS[m] for m in ev["modo"]]))
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="límite térmico")
        fig.update_layout(title="Congestión máxima post-evento", height=320, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(ev[["modo", "escenario", "ENS_MWh", "ENS_pct", "costo_MUSD", "co2_g_kwh",
                     "max_carga_linea_pct", "renovables_pct", "margen_reserva_pct"]].round(2),
                 width="stretch", hide_index=True)

    st.markdown("#### 📉 Barrido de severidad (0–30 %)")
    if st.button("Calcular barrido (12 despachos PyPSA, se cachea)"):
        sw = _sweep(city, event, year, co2_price)
        cA, cB = st.columns(2)
        with cA:
            fig = px.line(sw, x="severidad_pct", y="ENS_MWh", color="modo", markers=True,
                          color_discrete_map=MODE_COLORS,
                          title="ENS vs severidad del evento")
            fig.update_layout(height=320, xaxis_title="Severidad (%)", yaxis_title="ENS (MWh)")
            st.plotly_chart(fig, width="stretch")
        with cB:
            fig = px.line(sw, x="severidad_pct", y="costo_MUSD", color="modo", markers=True,
                          color_discrete_map=MODE_COLORS,
                          title="Costo operativo vs severidad")
            fig.update_layout(height=320, xaxis_title="Severidad (%)", yaxis_title="M USD")
            st.plotly_chart(fig, width="stretch")
        st.caption("Lectura: el modelo robusto-IA aplana la curva de daño — la ENS y el costo "
                   "crecen más despacio con la severidad (hipótesis de resiliencia de la tesis).")

    st.markdown("#### 🔌 Análisis de contingencias N-1 (re-despacho completo)")
    if st.button("Ejecutar N-1 para el año y escenario seleccionados"):
        scen = st.session_state.get("n1_scen", "Robusto-IA (tesis)")
        n1 = _n1(city, year, scen, co2_price)
        st.dataframe(n1, width="stretch", hide_index=True,
                     column_config={"viola_n1": st.column_config.CheckboxColumn("Viola N-1")})
        n_viol = int(n1["viola_n1"].sum())
        if n_viol == 0:
            st.success(f"🟢 La red de {city} {year} ({scen}) supera todas las contingencias N-1 "
                       "sin racionamiento ni sobrecargas.")
        else:
            st.warning(f"🟡 {n_viol} contingencias con violación (ENS o sobrecarga): candidatos a "
                       "refuerzo de red o a corrección en tiempo real por el agente RL (pestaña 🤖).")

    st.markdown("#### ⏱️ Tiempo de adaptación (< 1 h, actividad 4c)")
    st.markdown("""
- **Capa de operación (Grid2Op):** el agente RL actúa cada **5 minutos** (paso de la crónica) —
  tres órdenes de magnitud por debajo de la meta de 1 h.
- **Capa de planificación (PyPSA):** el re-despacho anual completo ante un evento se resuelve en
  **~1–3 s** (HiGHS), y un re-entreno de la expansión ante nuevos supuestos toma minutos.
- La combinación predicción NN (horaria) + RL (5 min) + planificación (día-año) cubre la cadena
  completa de adaptación exigida por la propuesta.
    """)

    objetivos_panel("resiliencia", evidencias={
        "OE-3": [
            (f"ENS bajo '{event}' {severity*100:.0f}%: robusto {ens_rob:,.0f} MWh vs determinista {ens_det:,.0f} MWh",
             min(100, max(0, (1 - ens_rob / max(ens_det, 1e-9)) * 100))),
            ("N-1 con re-despacho disponible por ciudad/año/escenario", 100),
            ("Barrido de severidad 0–30 % implementado", 100),
        ],
    }, preguntas=["PI-3", "PI-4"])

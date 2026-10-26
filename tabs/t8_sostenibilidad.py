# -*- coding: utf-8 -*-
"""Pestaña 8 · 🌱 Sostenibilidad e impacto — OE-4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import sustainability as sus
from core import pypsa_planning as pp
from core.config import METAS
from core.ui_helpers import page_intro, objetivos_panel, kpi_row, SCENARIO_COLORS


@st.cache_data(show_spinner=False)
def _trajectory(city, scenario, co2_price, gas_shock):
    return sus.trajectory(city, scenario, co2_price=co2_price, gas_shock=gas_shock)


@st.cache_data(show_spinner=False)
def _metas(city, co2_price, gas_shock):
    return sus.metas_progress(city, co2_price, gas_shock)


@st.cache_data(show_spinner=False)
def _radar(city, year):
    return sus.radar_indices(city, year)


def render():
    page_intro(
        "sostenibilidad",
        "🌱 Sostenibilidad, costos e impacto de tecnologías futuras · OE-4",
        "Trayectorias 2026–2035 del balance anual (capacidad × factor de capacidad) por escenario, "
        "consistentes con las reglas de expansión de PyPSA pero ~1000× más rápidas — permiten "
        "comparar las **16 ciudades**. Se evalúan las metas cuantitativas de la propuesta "
        "(CO₂ −20 %, renovables +25 pp, OPEX −15 %, eficiencia +30 % vs BAU a 5 años) y el "
        "radar urbano de las tres dimensiones de variables (Demanda, Resiliencia, Sostenibilidad).",
    )

    cities = sd.list_cities()
    c1, c2, c3 = st.columns([2, 1, 1])
    city = c1.selectbox("Ciudad foco", cities, index=cities.index("Bogotá"), key="t8_city")
    co2_price = c2.number_input("Precio CO₂ (USD/t)", 0, 150, 30, 5, key="t8_co2")
    gas_shock = c3.number_input("Shock gas (fracción)", 0.0, 1.0, 0.0, 0.05, key="t8_gas")

    # --- trayectoria por escenario ---
    trajs = {sc: _trajectory(city, sc, co2_price, gas_shock) for sc in pp.ESCENARIOS}
    metrics = [("co2_g_kwh", "Intensidad de carbono (g CO₂/kWh)"),
               ("renovables_pct", "Participación renovable (%)"),
               ("opex_MUSD", "OPEX (M USD/año)"),
               ("eficiencia_idx", "Índice de eficiencia (0–100)")]
    cols = st.columns(4)
    for col, (k, title) in zip(cols, metrics):
        fig = go.Figure()
        for sc, tr in trajs.items():
            fig.add_trace(go.Scatter(x=tr.index, y=tr[k], name=sc,
                                     line=dict(color=SCENARIO_COLORS[sc], width=2)))
        fig.update_layout(title=title, height=240, margin=dict(t=40, b=10),
                          legend=dict(font=dict(size=8)))
        col.plotly_chart(fig, width="stretch")

    # inversión y déficit
    cA, cB = st.columns(2)
    with cA:
        inv = pd.DataFrame({sc: tr["inversion_MUSD"].cumsum() for sc, tr in trajs.items()})
        fig = px.area(inv, x=inv.index, y=inv.columns, color_discrete_map=SCENARIO_COLORS,
                      title="Inversión acumulada (M USD)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
    with cB:
        dem = pd.DataFrame({sc: tr["demanda_TWh"] for sc, tr in trajs.items()})
        ofr = pd.DataFrame({sc: tr["oferta_TWh"] for sc, tr in trajs.items()})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dem.index, y=dem["Robusto-IA (tesis)"], name="Demanda",
                                 line=dict(color="#073B4C", width=3)))
        for sc in pp.ESCENARIOS:
            fig.add_trace(go.Scatter(x=ofr.index, y=ofr[sc], name=f"Oferta · {sc}",
                                     line=dict(color=SCENARIO_COLORS[sc], width=1.5, dash="dot")))
        fig.update_layout(title="Demanda vs oferta (TWh)", height=300,
                          legend=dict(orientation="h", y=-0.25, font=dict(size=9)))
        st.plotly_chart(fig, width="stretch")

    # --- metas ---
    st.markdown("#### 🎯 Metas cuantitativas de la propuesta (Robusto-IA vs BAU, 2031)")
    m = _metas(city, co2_price, gas_shock)
    for _, row in m.iterrows():
        st.progress(min(1.0, row["avance_pct"] / 100),
                    text=f"{row['meta']} — {row['objetivo']} · logrado {row['logrado']} · avance {row['avance_pct']:.0f} %")
    st.caption("Las metas se miden frente al escenario BAU del mismo año: así se aísla el efecto "
               "del modelo robusto (respuesta a la hipótesis de la tesis).")

    # --- radar ---
    st.markdown("#### 🕸️ Radar urbano (dimensiones de la propuesta)")
    year_r = st.slider("Año del radar", 2026, 2035, 2030)
    rad = _radar(city, year_r)
    cA, cB = st.columns([1, 1.4])
    with cA:
        cats = ["Demanda", "Resiliencia", "Sostenibilidad"]
        vals = [rad[c] for c in cats] + [rad[cats[0]]]
        fig = go.Figure(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill="toself",
                                        line=dict(color="#06D6A0"), name=city))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=360,
                          title=f"Índices 0–100 · {city} {year_r}", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with cB:
        st.metric("Margen de reserva (plan robusto)", f"{rad['margen_reserva_pct']:.0f} %")
        st.markdown("""
- **Demanda**: presión de crecimiento poblacional + VE + data centers.
- **Resiliencia**: SAIDI (inverso), pérdidas y almacenamiento del plan robusto.
- **Sostenibilidad**: renovables e intensidad de carbono del plan.
        """)

    # --- comparación multi-ciudad ---
    st.markdown("#### 🌍 Comparación multi-ciudad (2031, Robusto-IA vs BAU)")
    if st.button("Calcular para las 16 ciudades (modelo anual rápido)"):
        rows = []
        for ct in cities:
            rob = sus.annual_balance(ct, 2031, "Robusto-IA (tesis)", co2_price, gas_shock)
            bau = sus.annual_balance(ct, 2031, "BAU tendencial", co2_price, gas_shock)
            rows.append(dict(ciudad=ct, region=sd.get_city(ct)["region"],
                             ren_robusto=rob["renovables_pct"],
                             co2_robusto=rob["co2_g_kwh"],
                             d_co2_pct=(rob["co2_Mt"] - bau["co2_Mt"]) / max(bau["co2_Mt"], 1e-9) * 100,
                             d_opex_pct=(rob["opex_MUSD"] - bau["opex_MUSD"]) / max(bau["opex_MUSD"], 1e-9) * 100))
        mc = pd.DataFrame(rows).sort_values("d_co2_pct")
        cA, cB = st.columns(2)
        with cA:
            fig = px.bar(mc, x="d_co2_pct", y="ciudad", color="region", orientation="h",
                         title="Δ CO₂ vs BAU en 2031 (%)", text=mc.d_co2_pct.round(1))
            fig.update_layout(height=480, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        with cB:
            fig = px.scatter(mc, x="co2_robusto", y="ren_robusto", color="region", size="ren_robusto",
                             text="ciudad", title="Mix 2031 (Robusto-IA): CO₂ vs renovables")
            fig.update_traces(textposition="top center")
            fig.update_layout(height=480, xaxis_title="g CO₂/kWh", yaxis_title="Renovables (%)")
            st.plotly_chart(fig, width="stretch")
        st.dataframe(mc.round(1), width="stretch", hide_index=True)
        cumple_co2 = int((mc.d_co2_pct <= -METAS["co2_menos_pct"]).sum())
        st.info(f"Ciudades que alcanzan la meta CO₂ (−{METAS['co2_menos_pct']:.0f} % vs BAU en 2031): "
                f"**{cumple_co2}/16**. Las de matriz fósil (Johannesburgo, Teherán, Singapur) son "
                "donde el modelo robusto genera mayor valor — y las intermedias africanas muestran "
                "el desafío de infraestructura de PI-6.")

    avg_av = float(m["avance_pct"].mean())
    objetivos_panel("sostenibilidad", evidencias={
        "OE-4": [
            ("Trayectorias CO₂/renovables/OPEX/eficiencia 2026–2035 por escenario", 100),
            (f"Avance medio hacia las metas de la propuesta ({city}): {avg_av:.0f} %", avg_av),
            ("Comparación multi-ciudad Robusto-IA vs BAU disponible", 100),
            ("Inserción de VE y data centers en la demanda proyectada", 100),
        ],
    }, preguntas=["PI-1", "PI-5", "PI-6"])

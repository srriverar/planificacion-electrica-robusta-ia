# -*- coding: utf-8 -*-
"""Pestaña 10 · 📈 Crecimiento poblacional, circunstancias y conclusiones — OE-1/3/4."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import population_scenarios as ps
from core.ui_helpers import page_intro, objetivos_panel, kpi_row


# ---- cache de cálculos pesados (el primer load computa factores de capacidad) ----
@st.cache_data(show_spinner="Proyectando 16 ciudades (primer cálculo ≈30-40 s)…")
def _assess(scen, circs, plan, year):
    return ps.assess_all(scen, tuple(circs), plan, year)


@st.cache_data(show_spinner=False)
def _traj(city, scen, circs, plan):
    return ps.trajectory(city, scen, tuple(circs), plan)


@st.cache_data(show_spinner=False)
def _global_concl(scen, circs, plan, year):
    return ps.global_conclusions(scen, tuple(circs), plan, year)


@st.cache_data(show_spinner=False)
def _city_concl(city, scen, circs, plan, year):
    return ps.city_conclusion(city, scen, tuple(circs), plan, year)


def _risk_colors(df: pd.DataFrame) -> dict:
    return {r.Ciudad: r._color for r in df.itertuples()}


def render():
    page_intro(
        "poblacion",
        "📈 Crecimiento poblacional, circunstancias y conclusiones · OE-1 / OE-3 / OE-4",
        "¿Qué pasa con la planificación si la población crece más rápido, si llega una ola "
        "migratoria, si se disparan los data centers o si hay una sequía renovable? Esta pestaña "
        "cruza **escenarios demográficos** × **circunstancias** × **plan de respuesta** y proyecta "
        "las **16 ciudades** hasta 2045, derivando **conclusiones** sobre déficit de capacidad, "
        "emisiones y riesgo. Es un modelo anual rápido coherente con las reglas de expansión de "
        "PyPSA; el despacho horario fino vive en las pestañas ⚡ y 🛡️.",
    )

    # ---------------- controles ----------------
    st.markdown("### 🎛️ Configura el escenario")
    c1, c2, c3, c4 = st.columns([1.6, 1.4, 1.4, 1.2])
    scen = c1.selectbox("Escenario demográfico", list(ps.SCENARIOS.keys()),
                        index=list(ps.SCENARIOS.keys()).index("Tendencia ONU"),
                        format_func=lambda s: f"{s} — {ps.SCENARIOS[s]['label']}",
                        key="pop_scen")
    plan = c2.selectbox("Plan de respuesta (oferta)", list(ps.PLANES.keys()),
                        index=list(ps.PLANES.keys()).index("Robusto-IA (tesis)"),
                        key="pop_plan")
    year = c3.selectbox("Año horizonte", [2030, 2035, 2040, 2045], index=1, key="pop_year")
    circs = c4.multiselect(
        "Circunstancias", list(ps.CIRCUNSTANCIAS.keys()),
        default=[], format_func=lambda k: ps.CIRCUNSTANCIAS[k]["label"], key="pop_circs")

    st.caption(f"**{scen}:** {ps.SCENARIOS[scen]['desc']}")
    if circs:
        st.caption(" · ".join(f"{ps.CIRCUNSTANCIAS[k]['label']} ({ps.CIRCUNSTANCIAS[k]['short']})"
                             for k in circs))

    # ---------------- conclusiones globales ----------------
    st.markdown("### 🧭 Conclusiones agregadas (16 ciudades)")
    concl = _global_concl(scen, circs, plan, year)
    for c in concl:
        st.markdown(f"- {c}")

    # ---------------- tabla + KPIs ----------------
    df = _assess(scen, circs, plan, year)
    n = len(df)
    criticos = int((df["Margen_pct"] < 0).sum())
    estables = int((df["Margen_pct"] >= 30).sum())
    cap_total = float(df["Nueva_cap_GW"].sum())
    pop_total = float(df["Pob_M"].sum())
    kpi_row([
        ("Población agregada", f"{pop_total:,.0f} M", f"{year}", "Suma de las 16 ciudades"),
        ("Ciudades en déficit", f"{criticos}/{n}", None, "Margen de capacidad < 0 %"),
        ("Ciudades estables", f"{estables}/{n}", None, "Margen ≥ 30 %"),
        ("Capacidad nueva requerida", f"{cap_total:,.0f} GW", None,
         "Brecha de adecuación de pico agregada"),
    ])

    st.markdown(f"**Panorama por ciudad en {year}** (ordenado por margen de capacidad, el más "
                "tensionado arriba):")
    disp = df.drop(columns=["_color"]).copy()
    styled = disp.style.format({
        "Pob_2026_M": "{:.2f}", "Pob_M": "{:.2f}", "dPob_pct": "{:+.0f}",
        "Demanda_TWh": "{:.1f}", "dDemanda_pct": "{:+.0f}", "Margen_pct": "{:.0f}",
        "Deficit_GWh": "{:.0f}", "Nueva_cap_GW": "{:.2f}", "Ren_pct": "{:.0f}", "CO2_Mt": "{:.1f}",
    }).background_gradient(subset=["Margen_pct"], cmap="RdYlGn")
    st.dataframe(styled, width="stretch", hide_index=True)

    # ---------------- visual: riesgo vs crecimiento ----------------
    vA, vB = st.columns([1.3, 1])
    with vA:
        fig = px.scatter(df, x="dPob_pct", y="Margen_pct", color="Riesgo", size="Demanda_TWh",
                         hover_name="Ciudad", size_max=42,
                         color_discrete_map={"Crítico · déficit": "#9D0208", "Riesgo alto": "#EF476F",
                                             "Vigilar": "#FFB703", "Estable": "#06D6A0"},
                         title=f"Crecimiento poblacional vs margen de capacidad ({year})")
        fig.add_hline(y=0, line_dash="dash", line_color="#9D0208", annotation_text="déficit")
        fig.add_hline(y=15, line_dash="dot", line_color="#FFB703", annotation_text="estrés 15 %")
        fig.update_layout(height=420, xaxis_title="Δ población 2026→%d (%%)" % year,
                          yaxis_title="margen de capacidad (%)")
        st.plotly_chart(fig, width="stretch")
    with vB:
        top = df.sort_values("Margen_pct").head(10).iloc[::-1]
        fig = px.bar(top, x="Margen_pct", y="Ciudad", orientation="h", color="Riesgo",
                     color_discrete_map={"Crítico · déficit": "#9D0208", "Riesgo alto": "#EF476F",
                                         "Vigilar": "#FFB703", "Estable": "#06D6A0"},
                     title="10 ciudades más tensionadas")
        fig.add_vline(x=0, line_dash="dash", line_color="#9D0208")
        fig.update_layout(height=420, showlegend=False, xaxis_title="margen (%)")
        st.plotly_chart(fig, width="stretch")

    # ---------------- detalle de una ciudad ----------------
    st.markdown("---")
    st.markdown("### 🔎 Proyección detallada de una ciudad")
    cities = sd.list_cities()
    d1, d2 = st.columns([2, 3])
    city = d1.selectbox("Ciudad", cities, index=cities.index("Bogotá") if "Bogotá" in cities else 0,
                        key="pop_city")
    tr = _traj(city, scen, circs, plan)
    concl_city = _city_concl(city, scen, circs, plan, year)
    d2.info(concl_city)

    gA, gB = st.columns(2)
    with gA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.index, y=tr["demanda_TWh"], name="Demanda",
                                 line=dict(color="#EF476F", width=3)))
        fig.add_trace(go.Scatter(x=tr.index, y=tr["oferta_TWh"], name="Oferta (plan)",
                                 fill="tozeroy", fillcolor="rgba(6,214,161,0.15)",
                                 line=dict(color="#06D6A0", width=2)))
        fig.add_trace(go.Scatter(x=tr.index, y=tr["demanda_dc_TWh"], name="Data centers",
                                 line=dict(color="#FFB703", width=1.5, dash="dot")))
        fig.update_layout(title=f"{city} · demanda vs oferta (TWh)", height=340,
                          xaxis_title="año", yaxis_title="TWh", margin=dict(t=44))
        st.plotly_chart(fig, width="stretch")
    with gB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.index, y=tr["margen_pct"], name="Margen capacidad",
                                 line=dict(color="#118AB2", width=3)))
        fig.add_trace(go.Scatter(x=tr.index, y=tr["pop_m"], name="Población (M)", yaxis="y2",
                                 line=dict(color="#8D99AE", width=2, dash="dot")))
        fig.add_hline(y=0, line_dash="dash", line_color="#9D0208")
        fig.add_hline(y=15, line_dash="dot", line_color="#FFB703")
        fig.update_layout(title=f"{city} · margen de capacidad y población", height=340,
                          xaxis_title="año", yaxis_title="margen (%)",
                          yaxis2=dict(overlaying="y", side="right", title="población (M)",
                                      showgrid=False), margin=dict(t=44))
        st.plotly_chart(fig, width="stretch")

    cA, cB, cC = st.columns(3)
    with cA:
        fig = px.line(tr, x=tr.index, y="renovables_pct", markers=True,
                      title="Participación renovable (%)")
        fig.update_layout(height=280); st.plotly_chart(fig, width="stretch")
    with cB:
        fig = px.line(tr, x=tr.index, y="co2_Mt", markers=True, title="CO₂ efectivo (Mt)")
        fig.update_layout(height=280); st.plotly_chart(fig, width="stretch")
    with cC:
        fig = px.line(tr, x=tr.index, y="pico_MW", markers=True, title="Pico de demanda (MW)")
        fig.update_layout(height=280); st.plotly_chart(fig, width="stretch")

    # ---------------- conclusiones por ciudad (todas) ----------------
    with st.expander("📄 Conclusiones ciudad por ciudad (las 16)"):
        if st.button("Generar conclusiones de las 16 ciudades", key="pop_gen_all"):
            st.session_state["pop_show_all"] = True
        if st.session_state.get("pop_show_all"):
            with st.spinner("Redactando conclusiones…"):
                for cty in cities:
                    st.markdown(f"- {_city_concl(cty, scen, circs, plan, year)}")
            st.caption("Conclusiones generadas con el modelo anual de escenarios "
                       "(core/population_scenarios.py) a partir de datos sintéticos.")

    # ---------------- lectura de robustez (comparación de planes) ----------------
    with st.expander("⚖️ ¿Cómo cambia el riesgo según el plan de respuesta?"):
        comp = {}
        for pl in ps.PLANES:
            d = _assess(scen, circs, pl, year)
            comp[pl] = int((d["Margen_pct"] < 15).sum())
        comp_df = pd.DataFrame({"Plan": list(comp.keys()),
                                "Ciudades en estrés/déficit (<15 % margen)": list(comp.values())})
        st.dataframe(comp_df, width="stretch", hide_index=True)
        fig = px.bar(comp_df, x="Plan", y="Ciudades en estrés/déficit (<15 % margen)",
                     color="Plan", text_auto=True,
                     title=f"Ciudades tensionadas en {year} según el plan (mismo escenario demográfico)")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("El plan **Robusto-IA (tesis)** suele sostener más ciudades fuera del estrés "
                   "que el BAU: evidencia del valor de la planificación robusta (OE-3/OE-4).")

    # ---------------- objetivos ----------------
    n_def = int((df["Margen_pct"] < 0).sum())
    objetivos_panel("poblacion", evidencias={
        "OE-1": [
            (f"Proyección demográfica y de demanda de las {n} ciudades (escenarios + circunstancias)", 100),
            ("Minería de patrones de crecimiento vinculada a la pestaña 🌍", 80),
        ],
        "OE-3": [
            (f"Identificación de déficit de adecuación en {n_def}/{n} ciudades bajo el escenario", 100),
            ("Comparación de planes de respuesta (BAU vs Transición vs Robusto-IA)", 100),
        ],
        "OE-4": [
            ("Trayectorias de CO₂ efectivo y % renovable bajo crecimiento poblacional", 100),
            (f"Capacidad nueva requerida agregada ≈{cap_total:,.0f} GW para {year}", 90),
            ("Vínculo con hiperdatacenters (Vessim) y metas de sostenibilidad", 85),
        ],
    }, preguntas=["PI-1", "PI-3", "PI-5"])

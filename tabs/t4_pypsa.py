# -*- coding: utf-8 -*-
"""Pestaña 4 · ⚡ Planificación con PyPSA — OE-3/OE-4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import pypsa_planning as pp
from core.ui_helpers import (page_intro, objetivos_panel, kpi_row, TECH_COLORS,
                             TECH_LABEL, SCENARIO_COLORS)

YEARS = tuple(range(2026, 2036))
# layout fijo del diagrama unifilar simplificado
BUS_XY = {"Interconexión": (0.5, 2.2), "Centro": (0.5, 1.2), "Norte": (0.5, 0.2),
          "Este": (1.4, 0.7), "Sur": (0.5, -0.8), "Oeste": (-0.4, 0.7),
          "Industrial": (1.4, 1.7)}


@st.cache_data(show_spinner="Resolviendo despacho óptimo (PyPSA + HiGHS)…")
def _run_year(city, year, scenario, co2_price, gas_shock):
    return pp.run_year(city, year, scenario, co2_price, gas_shock)


@st.cache_data(show_spinner="Resolviendo trayectorias de expansión 2026–2035 (LP por año)…")
def _trajectory(city, scenario, co2_price, gas_shock):
    return pp.expansion_trajectory(city, YEARS, scenario, co2_price, gas_shock)


@st.cache_data(show_spinner=False)
def _caps_by_year(city, scenario, co2_price, gas_shock):
    rows = []
    for y in YEARS:
        caps = pp.capacities_year(city, y, scenario, co2_price, gas_shock)
        for t, v in caps.items():
            if not t.startswith("_") and v > 0:
                rows.append(dict(year=y, tech=t, mw=v))
    return pd.DataFrame(rows)


def _network_figure(city, year, scenario, co2_price):
    import pypsa
    n = pp.build_network(city, year, scenario, co2_price)
    fig = go.Figure()
    for _, l in n.lines.iterrows():
        x0, y0 = BUS_XY[l.bus0]; x1, y1 = BUS_XY[l.bus1]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color="#ADB5BD", width=2), hoverinfo="skip",
                                 showlegend=False))
    for bus, (x, y) in BUS_XY.items():
        color = "#118AB2" if "Interconexión" in bus or bus == "Centro" else "#073B4C"
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[bus],
                                 textposition="top center", marker=dict(size=16, color=color),
                                 showlegend=False, hoverinfo="text"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      title=f"Red sintética agregada · {city} {year}")
    return fig


def render():
    page_intro(
        "pypsa",
        "⚡ Planificación de la expansión y despacho óptimo con PyPSA · OE-3 / OE-4",
        "Red sintética agregada de 7 buses por ciudad (interconexión 220 kV, centro 220 kV y "
        "5 zonas 110 kV). **Despacho óptimo DC (LOPF)** sobre 96 instantes representativos "
        "(4 estaciones × 24 h) y **trayectoria de expansión 2026–2035** para tres escenarios "
        "de planificación. El racionamiento tiene costo explícito (VOLL 500 USD/MWh), así la "
        "energía no suministrada entra al óptimo.",
    )

    cities = sd.list_cities()
    c1, c2, c3, c4 = st.columns([2, 1.4, 1, 1])
    city = c1.selectbox("Ciudad", cities, index=cities.index("Bogotá"), key="t4_city")
    scenario = c2.selectbox("Escenario de planificación", pp.ESCENARIOS, index=2, key="t4_scen")
    year = c3.slider("Año de estudio", 2026, 2035, 2030)
    co2_price = c4.number_input("Precio CO₂ (USD/t)", 0, 150, 30, 5, key="t4_co2")
    gas_shock = st.slider("Shock de precio del gas (fracción, ej. 0.3 = +30 %)", 0.0, 1.0, 0.0, 0.05,
                          help="Prueba de robustez económica (optimización robusta de la propuesta)")

    r = _run_year(city, year, scenario, co2_price, gas_shock)
    kpi_row([
        ("Costo operativo anual", f"{r['objective_usd']/1e6:,.0f} M USD", None, "Objective del LP: combustible + O&M + CO2 + racionamiento"),
        ("Participación renovable", f"{r['renewable_share_pct']:.1f} %", None),
        ("Intensidad de carbono", f"{r['co2_g_per_kwh']:,.0f} g CO₂/kWh", None),
        ("Energía no suministrada", f"{r['ens_mwh']:,.0f} MWh", None, "Racionamiento óptimo (VOLL 500 USD/MWh)"),
    ])
    kpi_row([
        ("Margen de reserva", f"{r['reserve_margin_pct']:.0f} %", None, "Capacidad firme + crédito renovable/almacenamiento sobre el pico"),
        ("Pico de demanda", f"{r['peak_mw']:,.0f} MW", None),
        ("Inversión del año", f"{r['investment_usd']/1e6:,.0f} M USD", None, "Δ capacidades × CAPEX sintético"),
        ("Carga máxima de línea", f"{r['line_max_loading_pct']:.0f} %", None, "|flujo|/s_nom en la red"),
    ])

    tA, tB = st.columns([1.1, 1])
    with tA:
        # mezcla de generación (donut)
        eg = r["energy_by_gen"]
        eg = eg[eg > 0]
        labels = [TECH_LABEL.get(k.replace("gen_", ""), ("Racionamiento" if k.startswith("racion") else k))
                  for k in eg.index]
        colors = [TECH_COLORS.get(k.replace("gen_", "").split("_")[0] if k.startswith("gen_") else "racionamiento", "#888")
                  for k in eg.index]
        fig = go.Figure(go.Pie(labels=labels, values=(eg / 1e6).round(2), hole=0.55,
                               marker_colors=colors, textinfo="label+percent"))
        fig.update_layout(title=f"Generación anual por tecnología (TWh) · {city} {year}", height=380)
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(_network_figure(city, year, scenario, co2_price), width="stretch")
    with tB:
        # despacho de un día representativo por estación
        season = st.selectbox("Día representativo (estación)", list(pp.SEASONS.keys()), index=2, key="t4_season")
        gp = r.get("gen_p")
        if gp is not None and len(gp):
            try:
                idx = pd.date_range(f"{year}-{pp.SEASONS[season]:02d}-15", periods=24, freq="h")
                snaps = gp.loc[idx]
            except Exception:
                snaps = gp.iloc[:24]
            disp = pd.DataFrame({
                (TECH_LABEL.get(c.replace("gen_", ""), "Racionamiento") if not c.startswith("racion")
                 else f"Racionamiento {c.split('_')[-1]}"): v for c, v in snaps.items()})
            # agrupar racionamientos
            rac = [c for c in disp.columns if c.startswith("Racionamiento ")]
            if rac:
                disp["Racionamiento (ENS)"] = disp[rac].sum(axis=1)
                disp = disp.drop(columns=rac)
            fig = px.area(disp, x=disp.index, y=disp.columns,
                          color_discrete_map={**{TECH_LABEL.get(k, k): v for k, v in TECH_COLORS.items()},
                                              "Racionamiento (ENS)": "#EF476F"})
            fig.update_layout(title=f"Despacho horario · día 15 de {season} ({year})",
                              yaxis_title="MW", height=420, legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, width="stretch")
        # precios marginales y carga de líneas
        prices = r["prices"]
        if prices is not None and "Centro" in prices.columns:
            idx = pd.date_range(f"{year}-{pp.SEASONS[season]:02d}-15", periods=24, freq="h")
            fig = go.Figure(go.Scatter(x=idx, y=prices.loc[idx, "Centro"],
                                       line=dict(color="#F18F01", width=2)))
            fig.update_layout(title=f"Precio marginal en Centro · {season} (USD/MWh)", height=220,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📈 Trayectoria de expansión 2026–2035 (3 escenarios)")
    show_traj = st.checkbox("Calcular trayectorias completas (resuelve 30 LP; se cachea)", value=False)
    if show_traj:
        trajs = {}
        for sc in pp.ESCENARIOS:
            trajs[sc] = _trajectory(city, sc, co2_price, gas_shock)
        m = st.columns(4)
        specs = [("costo_MUSD", "Costo operativo (M USD/año)"), ("co2_g_kwh", "Intensidad CO₂ (g/kWh)"),
                 ("ren_pct", "Renovables (%)"), ("ENS_MWh", "Energía no suministrada (MWh)")]
        for col, (k, title) in zip(m, specs):
            fig = go.Figure()
            for sc, tr in trajs.items():
                fig.add_trace(go.Scatter(x=tr.index, y=tr[k], name=sc,
                                         line=dict(color=SCENARIO_COLORS[sc], width=2)))
            fig.update_layout(title=title, height=260, margin=dict(t=40, b=10),
                              legend=dict(font=dict(size=9)))
            col.plotly_chart(fig, width="stretch")
        inv = pd.DataFrame({sc: tr["inversion_MUSD"].cumsum() for sc, tr in trajs.items()})
        fig = px.area(inv, x=inv.index, y=inv.columns,
                      color_discrete_map=SCENARIO_COLORS,
                      title="Inversión acumulada en generación (M USD)")
        fig.update_layout(height=320, yaxis_title="M USD acumulados")
        st.plotly_chart(fig, width="stretch")

    caps = _caps_by_year(city, scenario, co2_price, gas_shock)
    piv = caps.pivot_table(index="year", columns="tech", values="mw", aggfunc="sum").fillna(0)
    fig = px.area(piv, x=piv.index, y=piv.columns,
                  color_discrete_map=TECH_COLORS,
                  title=f"Evolución de capacidad instalada (MW) · {scenario}")
    fig.update_layout(height=360, yaxis_title="MW", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    with st.expander("🔍 Detalle del despacho (tablas)"):
        st.dataframe((r["energy_by_gen"] / 1000).round(1).rename("GWh/año").to_frame(),
                     width="stretch")
        st.dataframe(r["line_loading"].round(1).rename("Carga máx. (%)").to_frame(),
                     width="stretch")
        cur = r.get("curtailment", {})
        if cur:
            st.write(f"Curtailment solar: **{cur.get('solar',0)/1000:,.1f} GWh** · "
                     f"eólico: **{cur.get('eolica',0)/1000:,.1f} GWh** (indicador de integración renovable, OE-3)")

    ens_ok = r["ens_mwh"] < max(r["total_load_mwh"] * 0.0005, 1)
    objetivos_panel("pypsa", evidencias={
        "OE-3": [
            ("Despacho óptimo resuelto con PyPSA/HiGHS para la ciudad y año seleccionados", 100),
            (f"ENS = {r['ens_mwh']:,.0f} MWh ({100*r['ens_mwh']/max(r['total_load_mwh'],1):.3f} % de la demanda)",
             100 if ens_ok else 60),
            (f"Integración renovable {r['renewable_share_pct']:.0f} % en el despacho", None),
        ],
        "OE-4": [
            (f"Costos de operación {r['objective_usd']/1e6:,.0f} M USD/año y CO₂ {r['co2_g_per_kwh']:,.0f} g/kWh cuantificados", 100),
            ("Trayectoria de expansión 2026–2035 por escenario disponible", 100),
            ("Inserción de tecnologías futuras (VE y data centers) incluida en la demanda proyectada", 100),
        ],
    }, preguntas=["PI-1", "PI-3", "PI-5"])

# -*- coding: utf-8 -*-
"""Pestaña 2 · 🌍 Ciudades y Datos — OE-1: minería de datos y aprendizaje no supervisado."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import forecasting as fc
from core.ui_helpers import page_intro, objetivos_panel, kpi_row, REGION_COLORS


@st.cache_data(show_spinner=False)
def _hourly(city: str, years: int = 2):
    return sd.city_hourly(city, years=years)


@st.cache_data(show_spinner="Entrenando K-Means sobre perfiles diarios…")
def _clusters(city: str, k: int = 4):
    return fc.cluster_daily_profiles(city, k=k)


@st.cache_data(show_spinner=False)
def _pca():
    return fc.pca_cities()


@st.cache_data(show_spinner=False)
def _summary():
    return sd.city_summary_table()


def render():
    page_intro(
        "ciudades",
        "🌍 Ciudades y Datos · Objetivo Específico 1",
        "Catálogo mundial de **16 ciudades cuya población está hoy entre 6 y 9 millones de "
        "habitantes** (capitales e intermedias), con series sintéticas horarias de demanda, "
        "clima, renovables, intensidad de carbono y precio. Sobre estos datos se aplican "
        "técnicas de **minería de datos y aprendizaje no supervisado** (K-Means y PCA) para "
        "identificar patrones complejos, como propone el OE-1 con caso de estudio Bogotá.",
    )

    summ = _summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciudades en catálogo", len(summ), help="Población 6–9 M hab. (valores ≈ reales 2025, resto sintético)")
    c2.metric("Población cubierta", f"{summ.poblacion_M.sum():.1f} M hab.")
    c3.metric("Demanda anual agregada", f"{summ['demanda_TWh_año'].sum():.0f} TWh")
    c4.metric("Data centers (pipeline)", f"{summ.datacenter_MW.sum():.0f} MW", help="Nueva tecnología que presiona la red (PI-1)")

    tab_mapa, tab_serie, tab_patrones, tab_pca, tab_calidad = st.tabs(
        ["🗺️ Mapa mundial", "📈 Serie urbana", "🧩 Patrones (K-Means)", "🔎 PCA de ciudades", "🧪 Calidad de datos"])

    with tab_mapa:
        metric = st.selectbox("Métrica del mapa", ["poblacion_M", "demanda_TWh_año", "pico_MW",
                                                   "renovables_pct", "ci_gCO2_kwh", "datacenter_MW",
                                                   "share_VE_pct", "saidi_h"], key="t2_metric")
        fig = px.scatter_geo(summ, lat="lat", lon="lon", size=metric, color="region",
                             hover_name="ciudad", size_max=38,
                             color_discrete_map=REGION_COLORS,
                             projection="natural earth",
                             title=f"Ciudades 6–9 M habitantes · {metric}")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summ.sort_values("poblacion_M", ascending=False),
                     width="stretch", hide_index=True)
        st.caption("Alcance poblacional: 'ciudad' (límites administrativos) o 'área metropolitana' "
                   "según la columna alcance_pob. Valores ≈ estimaciones 2025; todo lo energético es sintético.")

    with tab_serie:
        col_a, col_b = st.columns([2, 1])
        cities = list(summ.ciudad)
        default_i = cities.index("Bogotá") if "Bogotá" in cities else 0
        city = col_a.selectbox("Ciudad foco", cities, index=default_i, key="t2_focus")
        days = col_b.slider("Días a visualizar", 7, 90, 30)
        df = _hourly(city)
        win = df.iloc[-days * 24:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=win.index, y=win.demand_mw, name="Demanda total",
                                 line=dict(color="#118AB2", width=2)))
        fig.add_trace(go.Scatter(x=win.index, y=win.dc_mw, name="Data centers",
                                 fill="tonexty", line=dict(color="#EF476F", width=1)))
        fig.add_trace(go.Scatter(x=win.index, y=win.ev_mw, name="Vehículos eléctricos",
                                 fill="tonexty", line=dict(color="#06D6A0", width=1)))
        fig.update_layout(height=360, title=f"Demanda horaria · {city} (últimos {days} días)",
                          yaxis_title="MW", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

        m10 = sd.city_monthly_10y(city)
        cA, cB = st.columns(2)
        with cA:
            fig2 = px.area(m10, x=m10.index, y=["demand_mw_avg", "dc_mw", "ev_mw"],
                           title=f"Tendencia mensual 10 años · {city} (MW promedio)")
            fig2.update_layout(height=320, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig2, width="stretch")
        with cB:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df.index, y=df.temp_c, name="Temperatura °C",
                                      line=dict(color="#E76F51")))
            fig3.add_trace(go.Scatter(x=df.index, y=df.demand_mw / df.demand_mw.max() * 40,
                                      name="Demanda (escala 0–40 °C)", line=dict(color="#118AB2")))
            fig3.update_layout(height=320, title="Clima vs demanda (2 años)",
                               legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig3, width="stretch")

        k = kpi_row([
            ("Pico histórico", f"{df.demand_mw.max():,.0f} MW"),
            ("Media", f"{df.demand_mw.mean():,.0f} MW"),
            ("Factor de pico", f"{df.demand_mw.max()/df.demand_mw.mean():.2f}"),
            ("Crecimiento anual", f"{sd.get_city(city)['growth_pct']} %"),
        ])

    with tab_patrones:
        col_a, col_b = st.columns([2, 1])
        city = col_a.selectbox("Ciudad para clustering", list(summ.ciudad), index=0, key="cl_city")
        k = col_b.slider("Número de clústeres (k)", 2, 8, 4)
        cl = _clusters(city, k)
        cA, cB = st.columns(2)
        with cA:
            fig = go.Figure()
            for ci_, row in cl["centers"].iterrows():
                fig.add_trace(go.Scatter(x=row.index, y=row.values, mode="lines+markers",
                                         name=cl["names"][ci_]))
            fig.update_layout(title=f"Centroides de perfiles diarios · {city} (normalizados)",
                              xaxis_title="Hora", yaxis_title="Demanda relativa", height=380,
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
        with cB:
            fig = px.line(cl["inertia"], x=cl["inertia"].index, y=cl["inertia"].values,
                          markers=True, title="Codo de inercia (elección de k)")
            fig.update_layout(height=380, xaxis_title="k", yaxis_title="Inercia")
            st.plotly_chart(fig, width="stretch")
        st.markdown("**Patrones clave identificados (minería de datos, meta ≥ 5 familias):**")
        st.markdown("""
1. **Doble pico residencial** (mañana 6–8 h y noche 18–21 h) en días hábiles.
2. **Efecto fin de semana**: caída comercial/industrial y desplazamiento del pico.
3. **Sensibilidad térmica**: refrigeración (T > 22 °C) y calefacción eléctrica (T < 15 °C).
4. **Carga plana y creciente de data centers** (24/7, +18 % anual) — nueva tecnología (PI-1).
5. **Rampa nocturna de vehículos eléctricos** (18–23 h) acoplada al pico residencial.
6. **Estacionalidad renovable** (solar/eólica) que modula precio e intensidad de carbono.
        """)

    with tab_pca:
        p = _pca()
        cA, cB = st.columns([1.2, 1])
        with cA:
            fig = px.scatter(p["scores"], x="PC1", y="PC2", color="region",
                             color_discrete_map=REGION_COLORS, text=p["scores"].index,
                             title=f"PCA de indicadores urbanos · var. explicada "
                                   f"{p['explained'][0]*100:.0f}% + {p['explained'][1]*100:.0f}%")
            fig.update_traces(textposition="top center", marker_size=11)
            fig.update_layout(height=460)
            st.plotly_chart(fig, width="stretch")
        with cB:
            load = p["loadings"]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=load.index, x=load["PC1"], orientation="h", name="PC1"))
            fig.add_trace(go.Bar(y=load.index, x=load["PC2"], orientation="h", name="PC2"))
            fig.update_layout(barmode="group", height=460, title="Cargas (componentes)")
            st.plotly_chart(fig, width="stretch")
        st.caption("Aprendizaje no supervisado sobre 11 indicadores urbanos: el PC1 separa "
                   "ciudades por escala/intensidad energética y el PC2 por estrés de red y "
                   "descarbonización — insumo del OE-1 para tipologías capitales vs intermedias.")

    with tab_calidad:
        city = st.selectbox("Ciudad para reporte de calidad", list(summ.ciudad), index=0, key="dq_city")
        df = _hourly(city)
        rep = fc.data_quality_report(df)
        st.dataframe(rep, width="stretch", hide_index=True)
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr.round(2), text_auto=True, aspect="auto",
                        title=f"Correlaciones de variables · {city}",
                        color_continuous_scale="RdBu", zmin=-1, zmax=1)
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption("Pipeline de limpieza (documentado en S1.3–S1.4): detección de NaN, outliers "
                   "IQR×3 y verificación de rangos físicos antes de entrenar redes neuronales.")

    objetivos_panel("ciudades", evidencias={
        "OE-1": [
            ("Catálogo sintético 16/16 ciudades (6–9 M hab.)", 100),
            ("Series multidimensionales horarias (demanda, clima, red, VE, DC)", 100),
            ("Patrones clave detectados con no supervisado (meta ≥ 5): 6 familias", 100),
            ("Caso de estudio Bogotá integrado al catálogo", 100),
            ("Siguiente fase: reemplazar por datos reales (XM/UPME/IDEAM)", None),
        ],
    }, preguntas=["PI-1", "PI-6"])

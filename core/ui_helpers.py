# -*- coding: utf-8 -*-
"""Helpers de UI: paneles de supuestos y objetivos de la tesis, KPIs y colores."""
from __future__ import annotations

import streamlit as st
import plotly.io as pio

from .config import TAB_INFO, OBJETIVOS_ESPECIFICOS, PREGUNTAS_INVESTIGACION

pio.templates.default = "plotly_white"

TECH_COLORS = {
    "hidro": "#2E86AB", "gas": "#F18F01", "carbon": "#4A4A4A", "nuclear": "#9B5DE5",
    "solar": "#FFB703", "eolica": "#48CAE4", "geotermica": "#E76F51",
    "biomasa": "#8AB17D", "petroleo": "#7F5539", "importaciones": "#ADB5BD",
    "bateria": "#06D6A0", "racionamiento": "#EF476F",
}
TECH_LABEL = {
    "hidro": "Hidroeléctrica", "gas": "Gas natural", "carbon": "Carbón",
    "nuclear": "Nuclear", "solar": "Solar PV", "eolica": "Eólica",
    "geotermica": "Geotérmica", "biomasa": "Biomasa", "petroleo": "Petróleo/Diésel",
    "importaciones": "Importaciones", "bateria": "Baterías",
}
SCENARIO_COLORS = {"BAU tendencial": "#EF476F", "Transición moderada": "#FFB703",
                   "Robusto-IA (tesis)": "#06D6A0"}
REGION_COLORS = {
    "América del Sur": "#06D6A0", "América del Norte": "#118AB2", "Europa": "#9B5DE5",
    "Asia": "#EF476F", "África": "#FFB703",
}
AGENT_LABEL = {"donothing": "Do-Nothing (base)", "heuristica": "Heurística PowerLineSwitch",
               "dqn": "DQN tesis (NumPy)"}
AGENT_COLORS = {"donothing": "#ADB5BD", "heuristica": "#FFB703", "dqn": "#06D6A0"}


def tech_color(name: str) -> str:
    base = name.replace("gen_", "").split("_")[0]
    return TECH_COLORS.get(base, "#888888")


# ---------------------------------------------------------------------------
def page_intro(tab_key: str, titulo: str, intro_md: str):
    """Encabezado de pestaña: título, descripción y supuestos (siempre visibles)."""
    st.subheader(titulo)
    st.markdown(intro_md)
    info = TAB_INFO.get(tab_key, {})
    with st.expander("📋 Supuestos de esta pestaña", expanded=False):
        for s in info.get("supuestos", []):
            st.markdown(f"- {s}")


def objetivos_panel(tab_key: str, evidencias: dict | None = None,
                    preguntas: list[str] | None = None):
    """Panel '🎯 ¿Cómo avanza la tesis aquí?' con objetivos + evidencia viva.

    evidencias: {oe_key: [(texto, pct_0_a_100 | None), ...]}
    preguntas:  códigos PI-x relevantes en la pestaña.
    """
    info = TAB_INFO.get(tab_key, {})
    oes = info.get("objetivos", [])
    st.markdown("---")
    st.markdown("#### 🎯 ¿Cómo se van cumpliendo los objetivos de la tesis en esta pestaña?")
    ev = evidencias or {}
    if not oes:
        st.info("Esta pestaña es transversal: alimenta los cuatro objetivos específicos.")
        oes = list(OBJETIVOS_ESPECIFICOS.keys())
    oes = [o for o in oes if o in OBJETIVOS_ESPECIFICOS] or list(OBJETIVOS_ESPECIFICOS.keys())
    cols = st.columns(len(oes))
    for col, oe in zip(cols, oes):
        with col:
            st.markdown(f"**{oe}**")
            st.caption(OBJETIVOS_ESPECIFICOS[oe][:170] + "…")
            for texto, pct in ev.get(oe, []):
                if pct is None:
                    st.markdown(f"• {texto}", unsafe_allow_html=False)
                else:
                    st.progress(min(1.0, max(0.0, pct / 100.0)),
                                text=f"{texto} · {pct:.0f} %")
    if preguntas:
        pq = {c: t for c, t in PREGUNTAS_INVESTIGACION}
        st.markdown("**Preguntas de investigación que responde esta pestaña:**")
        for c in preguntas:
            if c in pq:
                st.markdown(f"- **{c}** — {pq[c]}")


def kpi_row(items: list[tuple], cols_per_row: int = 4):
    """items: (label, value, delta|None, help|None)."""
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[i:i + cols_per_row]):
            label, value = item[0], item[1]
            delta = item[2] if len(item) > 2 else None
            help_ = item[3] if len(item) > 3 else None
            col.metric(label, value, delta, help=help_)


def badge(cumple: bool) -> str:
    return "🟢 cumplido" if cumple else "🟡 en progreso"


def st_data_table(df, **kwargs):
    st.dataframe(df, width="stretch", **kwargs)

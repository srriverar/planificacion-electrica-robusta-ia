# -*- coding: utf-8 -*-
"""
⚡ Planificación Robusta de Sistemas Eléctricos Urbanos con IA
Dashboard interactivo de la tesis doctoral de Juan José Bernal Segura
(dirigida por el Prof. Sergio Raúl Rivera Rodríguez) — UNAL Sede Bogotá.

Ejecutar local:      streamlit run app.py
Desplegar:           Streamlit Community Cloud (ver pestaña 🎯 Objetivos)

Herramientas acopladas: PyPSA (planificación) · Grid2Op (refuerzo) ·
Vessim (data centers) · scikit-learn (NN/ensembles) · Streamlit.
Todos los datos son SINTÉTICOS y reproducibles (semillas fijas).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from core import config as cfg
from core.ui_helpers import pio
from tabs import (t1_inicio, t2_ciudades_datos, t3_prediccion, t4_pypsa,
                  t5_grid2op_rl, t6_vessim_dc, t7_resiliencia,
                  t8_sostenibilidad, t9_objetivos, t10_poblacion, t11_mapas)

pio.templates.default = "plotly_white"

st.set_page_config(page_title=cfg.APP_TITLE, page_icon=cfg.APP_ICON,
                   layout="wide", initial_sidebar_state="expanded")

# --- sidebar: créditos de diseñadores + navegación ---
t9_objetivos.sidebar_credits()
st.sidebar.markdown("### Pestañas")
st.sidebar.caption(
    "🏠 Inicio · 🌍 Ciudades y Datos · 🧠 Predicción · ⚡ PyPSA · 🤖 Grid2Op · "
    "🖥️ Vessim · 🛡️ Resiliencia · 🌱 Sostenibilidad · 📈 Crecimiento · 🗺️ Mapas de Red · 🎯 Objetivos"
)
with st.sidebar.expander("ℹ️ Sobre el dashboard"):
    st.markdown(f"**{cfg.APP_TITLE}**")
    st.caption(cfg.APP_SHORT)
    st.markdown("**Herramientas:** PyPSA · Grid2Op · Vessim · scikit-learn · Streamlit")
    st.markdown("**Caso de estudio:** Bogotá (catálogo mundial de 16 ciudades 6–9 M hab.)")

# --- encabezado global ---
st.title(f"{cfg.APP_ICON} {cfg.APP_TITLE}")
st.caption(f"{cfg.APP_SHORT} · Diseñadores: {cfg.DISENADORES['estudiante']['nombre']} "
           f"(estudiante) y {cfg.DISENADORES['profesor']['nombre']} (profesor director) · "
           f"Universidad Nacional de Colombia – Sede Bogotá")

tabs = st.tabs([
    "🏠 Inicio", "🌍 Ciudades y Datos", "🧠 Predicción", "⚡ PyPSA",
    "🤖 Grid2Op", "🖥️ Vessim", "🛡️ Resiliencia", "🌱 Sostenibilidad",
    "📈 Crecimiento", "🗺️ Mapas de Red", "🎯 Objetivos",
])

renderers = [
    t1_inicio.render, t2_ciudades_datos.render, t3_prediccion.render, t4_pypsa.render,
    t5_grid2op_rl.render, t6_vessim_dc.render, t7_resiliencia.render,
    t8_sostenibilidad.render, t10_poblacion.render, t11_mapas.render, t9_objetivos.render,
]

for tab, renderer in zip(tabs, renderers):
    with tab:
        try:
            renderer()
        except Exception as e:  # pragma: no cover - resiliencia del demo
            st.error(f"Error al renderizar esta pestaña: {type(e).__name__}: {e}")
            st.info("Revisa que hayas ejecutado los scripts de entrenamiento/evaluación "
                    "(ver pestaña 🎯 Objetivos) o reduce la carga computacional de los controles.")
            st.exception(e)

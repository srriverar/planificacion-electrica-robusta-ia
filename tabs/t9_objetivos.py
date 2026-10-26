# -*- coding: utf-8 -*-
"""Pestaña 9 · 🎯 Objetivos, supuestos consolidados y despliegue."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as cfg
from core import synthetic_data as sd
from core.ui_helpers import page_intro, badge


CRONOGRAMA_DEMO = [
    ("Mes 3", "Actividad 1 · Recopilar/procesar datos y detectar ≥5 patrones",
     "🌍 Ciudades y Datos", "🟢 Demo: 16 ciudades sintéticas + 6 familias de patrones (K-Means/PCA)"),
    ("Mes 12", "Actividad 2 · Modelo híbrido NN+RL (error ≤5 %, eficiencia 90 %)",
     "🧠 Predicción + 🤖 Grid2Op", "🟢 Demo: MAPE del ensemble y DQN entrenado operando la red proxy"),
    ("Mes 18", "Actividad 3 · Ensemble multi-modelo (+15 % precisión 5–10 años)",
     "🧠 Predicción", "🟡 Parcial: stacking Ridge demostrado a 24 h; horizonte multianual pendiente"),
    ("Mes 20", "Actividad 4 · Evaluación con 2 años de prueba (adaptación <1 h, +25 % renovables)",
     "🛡️ Resiliencia + ⚡ PyPSA", "🟢 Demo: N-1, barrido de severidad y expansión renovable por escenario"),
    ("Mes 22", "Actividad 5 · Impacto Bogotá (−20 % CO2, +30 % eficiencia, −15 % OPEX, 5 años)",
     "🌱 Sostenibilidad", "🟢 Demo: metas vs BAU 2026→2031 con avance porcentual por ciudad"),
    ("Mes 24", "Actividad 6 · Dashboard explicable (≥90 % decisiones explicadas, panel de expertos)",
     "Todo el dashboard", "🟢 Demo: supuestos + objetivos por pestaña, importancias, dependencia parcial, Q-values"),
]

MAPEO_PI = {
    "PI-1": ["🌍", "⚡", "🖥️", "📈", "🗺️"],
    "PI-2": ["🧠", "🤖"],
    "PI-3": ["🧠", "🛡️", "📈"],
    "PI-4": ["🛡️", "🤖", "🗺️"],
    "PI-5": ["⚡", "🖥️", "🧠", "📈"],
    "PI-6": ["🌍", "🤖", "🌱", "📈", "🗺️"],
}


def render():
    page_intro(
        "objetivos",
        "🎯 Objetivos de la tesis, supuestos consolidados y despliegue",
        "Registro maestro de **cómo cada pestaña alimenta los objetivos específicos**, el estado "
        "de las **metas cuantitativas** de la propuesta, los **supuestos consolidados** (S0–S10) y "
        "las instrucciones de **despliegue** (GitHub + Streamlit Community Cloud).",
    )

    st.markdown("### Objetivos específicos y dónde se materializan")
    rows = [
        dict(OE="OE-1", objetivo=cfg.OBJETIVOS_ESPECIFICOS["OE-1"],
             pestañas="🌍 Ciudades · 🧪 Calidad · 📈 Crecimiento · 🗺️ Mapas",
             estado="🟢 Demo completo con datos sintéticos (falta: datos reales Bogotá)"),
        dict(OE="OE-2", objetivo=cfg.OBJETIVOS_ESPECIFICOS["OE-2"],
             pestañas="🧠 Predicción · 🤖 Grid2Op · 🗺️ Mapas",
             estado="🟢 NN+ensemble y DQN operativos en las 16 ciudades (falta: red PyPSA específica)"),
        dict(OE="OE-3", objetivo=cfg.OBJETIVOS_ESPECIFICOS["OE-3"],
             pestañas="🧠 Precisión · ⚡ PyPSA · 🛡️ Resiliencia · 📈 Crec. · 🗺️ Mapas",
             estado="🟢 Métricas de desempeño, eficiencia, integración renovable y adecuación"),
        dict(OE="OE-4", objetivo=cfg.OBJETIVOS_ESPECIFICOS["OE-4"],
             pestañas="⚡ PyPSA · 🖥️ Vessim · 🌱 Sostenib. · 📈 Crec. · 🗺️ Mapas",
             estado="🟢 Sostenibilidad, costos, tecnologías futuras y escenarios de crecimiento"),
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=230)

    st.markdown("### Preguntas de investigación → pestañas")
    pi_rows = [dict(PI=c, pregunta=t, pestañas=" ".join(MAPEO_PI.get(c, [])))
               for c, t in cfg.PREGUNTAS_INVESTIGACION]
    st.dataframe(pd.DataFrame(pi_rows), width="stretch", hide_index=True)

    st.markdown("### Cronograma de la propuesta → estado en el demo")
    st.dataframe(pd.DataFrame(CRONOGRAMA_DEMO,
                              columns=["Hito", "Actividad (propuesta)", "Pestaña(s)", "Estado en el demo"]),
                 width="stretch", hide_index=True)

    st.markdown("### 📋 Supuestos consolidados (registro S0–S10)")
    for tab_key, info in cfg.TAB_INFO.items():
        if info["supuestos"]:
            with st.expander(f"Supuestos · pestaña `{tab_key}`"):
                for s in info["supuestos"]:
                    st.markdown(f"- {s}")

    st.markdown("### 🌐 Fuentes de datos reales para la siguiente fase")
    st.dataframe(pd.DataFrame(cfg.FUENTES_REALES_FUTURAS, columns=["Ámbito", "Fuente"]),
                 width="stretch", hide_index=True)

    st.markdown("### 🚀 Despliegue (GitHub + Streamlit Community Cloud)")
    st.markdown("""
**1. Subir a GitHub**
```bash
git init
git add .
git commit -m "Dashboard tesis: planificación robusta de sistemas eléctricos urbanos con IA"
git branch -M main
git remote add origin https://github.com/<TU_USUARIO>/planificacion-electrica-robusta-ia.git
git push -u origin main
```
**2. Desplegar en Streamlit Community Cloud**
1. Entra a https://share.streamlit.io con tu cuenta de GitHub.
2. *New app* → selecciona el repositorio, rama `main` y archivo `app.py`.
3. (Opcional) Entrena de nuevo el agente en tu máquina y sube `models/dqn_case5.npz`
   y `results/rl_eval.json` para resultados instantáneos en la nube.
4. *Deploy*. La app instala `requirements.txt` automáticamente.

> **Nota Grid2Op:** el ambiente `rte_case5_example` viene incluido en el paquete, así la app
> funciona en la nube sin descargas. Si quieres ambientes L2RPN mayores, descárgalos localmente.
""")
    st.caption(cfg.FOOTER)


def sidebar_credits():
    st.sidebar.markdown("### Diseñadores")
    e = cfg.DISENADORES["estudiante"]; p = cfg.DISENADORES["profesor"]
    st.sidebar.markdown(f"**🎓 {e['nombre']}**\n\n{e['rol']}")
    st.sidebar.markdown(f"**👨‍🏫 {p['nombre']}**\n\n{p['rol']}")
    st.sidebar.caption("Universidad Nacional de Colombia – Sede Bogotá")
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Datos 100 % sintéticos con fines de demostración metodológica. "
                       "Poblaciones ≈ estimaciones reales 2025 (6–9 M hab.).")

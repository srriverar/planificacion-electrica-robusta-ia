# -*- coding: utf-8 -*-
"""Pestaña 1 · Inicio: tesis, diseñadores, arquitectura del modelo y tracker global."""
from __future__ import annotations

import os
import streamlit as st
import pandas as pd

from core import config as cfg
from core import synthetic_data as sd
from core.ui_helpers import page_intro, kpi_row, badge


def _rl_summary() -> dict | None:
    from core.grid2op_rl import load_eval
    ev = load_eval()
    if not ev:
        return None
    df = pd.DataFrame(ev["rows"])
    out = {}
    for ag, g in df.groupby("agent"):
        out[ag] = dict(episodios=len(g), alive_promedio=float(g["steps_alive"].mean()),
                       alive_total=int(g["steps_alive"].sum()),
                       reward_total=float(g["reward_total"].sum()),
                       sobrevive_todo=int(g["survived_all"].sum()))
    return out


def _global_tracker() -> pd.DataFrame:
    from core import sustainability as sus
    rl = _rl_summary()
    # OE-1: activos demo de datos
    n_cities = len(sd.list_cities())
    oe1 = min(100, round(n_cities / 16 * 100))
    # OE-2: agente RL entrenado (proxy Grid2Op) + modelos NN disponibles en pestaña 3
    if rl and "dqn" in rl and "donothing" in rl:
        ratio = rl["dqn"]["alive_total"] / max(rl["donothing"]["alive_total"], 1)
        oe2 = min(100, round(35 + min(ratio, 4) / 4 * 65))
    else:
        oe2 = 20
    # OE-3 / OE-4: avance de metas en ciudad caso de estudio (modelo anual rápido)
    try:
        m = sus.metas_progress("Bogotá")
        oe34 = round(float(m["avance_pct"].mean()))
        oe4 = round(float(m.loc[m["meta"].str.contains("costos|Eficiencia"), "avance_pct"].mean()))
    except Exception:
        oe34 = oe4 = 0
    rows = [
        dict(objetivo="OE-1 · Datos y patrones (minería + no supervisado)",
             evidencia_demo=f"{n_cities}/16 ciudades sintéticas 6–9 M hab.; clustering K-Means y PCA operativos en la pestaña 🌍",
             avance_pct=oe1),
        dict(objetivo="OE-2 · Modelo híbrido con aprendizaje por refuerzo",
             evidencia_demo=("DQN NumPy entrenado en Grid2Op: sobrevive "
                             f"{rl['dqn']['sobrevive_todo']}/{rl['dqn']['episodios']} crónicas completas "
                             f"(Do-Nothing {rl['donothing']['sobrevive_todo']}/{rl['donothing']['episodios']})"
                             if rl and "dqn" in rl else "Entrenamiento del DQN en curso / pendiente"),
             avance_pct=oe2),
        dict(objetivo="OE-3 · Desempeño: precisión, eficiencia e integración renovable",
             evidencia_demo="MAPE del ensemble NN (pestaña 🧠) + expansión PyPSA con renovables (pestaña ⚡)",
             avance_pct=oe34),
        dict(objetivo="OE-4 · Impacto: sostenibilidad, eficiencia y costos con nuevas tecnologías",
             evidencia_demo="Metas CO2/OPEX/eficiencia (pestaña 🌱) + data centers Vessim (pestaña 🖥️)",
             avance_pct=oe4),
    ]
    return pd.DataFrame(rows)


def render():
    page_intro(
        "inicio",
        "⚡ Modelo robusto de planificación de sistemas eléctricos urbanos con inteligencia computacional",
        "Dashboard interactivo de demostración de la metodología de la tesis doctoral. "
        "Cada pestaña materializa una etapa del modelo —datos sintéticos, predicción con redes "
        "neuronales, planificación PyPSA, operación con refuerzo Grid2Op y data centers Vessim— "
        "e informa **sus supuestos** y **el avance de los objetivos de la tesis**.",
    )

    st.markdown(f"### {cfg.TITULO_TESIS}")
    c1, c2 = st.columns(2)
    with c1:
        d = cfg.DISENADORES["estudiante"]
        st.info(f"**Diseñador (estudiante):** {d['nombre']}\n\n{d['rol']}\n\n{d['institucion']}")
    with c2:
        p = cfg.DISENADORES["profesor"]
        st.info(f"**Diseñador (profesor director):** {p['nombre']}\n\n{p['rol']}\n\n{p['institucion']}")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### Objetivo general")
        st.markdown(f"> {cfg.OBJETIVO_GENERAL}")
        st.markdown("#### Hipótesis")
        st.caption(cfg.HIPOTESIS)
    with c2:
        st.markdown("#### El demo en cifras")
        rl = _rl_summary()
        kpi_row([
            ("Ciudades sintéticas 6–9 M", len(sd.list_cities()), None, "Poblaciones ≈ reales 2025; datos energéticos sintéticos"),
            ("Herramientas acopladas", "PyPSA · Grid2Op · Vessim", None, "Más scikit-learn y Streamlit"),
            ("Agente RL (crónicas completas)", f"{rl['dqn']['sobrevive_todo']}/{rl['dqn']['episodios']}" if rl and 'dqn' in rl else "—",
             None, "DQN NumPy pre-entrenado sobre Grid2Op rte_case5_example"),
        ], cols_per_row=3)

    st.markdown("#### Arquitectura del modelo propuesto")
    st.graphviz_chart("""
    digraph {
      graph [rankdir=LR, bgcolor="transparent", fontname="helvetica"]
      node [shape=box, style="rounded,filled", fontname="helvetica", fontsize=11]
      A [label="🌍 Datos sintéticos\n16 ciudades 6–9 M hab.\ndemanda · clima · red · VE · DC", fillcolor="#E8F4F8"]
      B [label="🧠 Inteligencia computacional\nK-Means · PCA · MLP (NN)\nGradient Boosting · Ensemble", fillcolor="#FDEBD0"]
      C [label="⚡ Planificación PyPSA\nexpansión 2026–2035\ndespacho óptimo · N-1", fillcolor="#E9F7EF"]
      D [label="🤖 Operación Grid2Op\nDQN (aprendizaje por refuerzo)\ncongestión · contingencias", fillcolor="#F5EEF8"]
      E [label="🖥️ Data centers Vessim\ncarga IA · PV · baterías\ndespacho carbon-aware", fillcolor="#FDEDEC"]
      F [label="🛡️ Resiliencia\neventos extremos\nrobusto vs determinista", fillcolor="#FEF9E7"]
      G [label="🌱 Sostenibilidad\nCO2 · renovables · OPEX\neficiencia · metas", fillcolor="#EAF7EC"]
      A -> B -> C
      B -> D
      A -> E
      C -> F
      D -> F
      E -> F
      C -> G
      E -> G
      F -> G
      G -> H [label="🎯 Objetivos OE-1…OE-4\n(tablero de avance)", fillcolor="#D6EAF8"]
    }
    """, width="stretch")

    st.markdown("#### Herramientas y su papel en la tesis")
    st.dataframe(pd.DataFrame(cfg.HERRAMIENTAS, columns=["Herramienta", "Versión", "Rol en el modelo"]),
                 width="stretch", hide_index=True)

    st.markdown("#### 🎯 Tablero global de avance de los objetivos (evidencia del demo)")
    tr = _global_tracker()
    for _, r in tr.iterrows():
        st.progress(min(1.0, r["avance_pct"] / 100), text=f"{r['objetivo']} — {r['avance_pct']} %")
        st.caption(r["evidencia_demo"])
    st.caption("El avance se calcula con evidencias del propio demo sobre datos sintéticos; "
               "la validación con datos reales de Bogotá (XM/UPME/IDEAM) es la fase siguiente del cronograma.")

    with st.expander("🧭 Guía rápida de pestañas"):
        st.markdown("""
| Pestaña | Qué demuestra | Objetivos |
|---|---|---|
| 🌍 Ciudades y Datos | Catálogo mundial 6–9 M hab., minería de datos y patrones no supervisados | OE-1 |
| 🧠 Predicción NN | Redes neuronales + ensemble vs baselines, bandas de incertidumbre, interpretabilidad | OE-2, OE-3 |
| ⚡ Planificación PyPSA | Expansión 2026–2035, despacho óptimo, costos, CO2, inversión por escenario | OE-3, OE-4 |
| 🤖 Grid2Op por ciudad | DQN entrenado vs Do-Nothing en un ambiente propio por ciudad (crónicas desde su demanda) | OE-2 |
| 🖥️ Data Centers Vessim | Co-simulación de hiperdatacenter: PV, baterías, despacho carbon-aware | OE-4, PI-1 |
| 🛡️ Resiliencia | Eventos extremos y N-1: plan determinista vs robusto-IA | OE-3, PI-4 |
| 🌱 Sostenibilidad | Metas CO2/renovables/OPEX/eficiencia y radar urbano | OE-4 |
| 📈 Crecimiento y escenarios | Proyección demográfica 2026–2045, circunstancias y conclusiones por ciudad | OE-1, OE-3, OE-4 |
| 🗺️ Mapas de Red (grid2poster) | Mapa OSM real por ciudad en 3 subpestañas: 🔌 transmisión + Grid2Op + crecimiento, 🏭 centrales (MW por fuente) y ☀️ glow solar (EarthPV / OSM+GPPD) | OE-1…OE-4 |
| 🎯 Objetivos y Supuestos | Registro consolidado de supuestos, metas y despliegue | Todos |
        """)

    st.markdown("#### ❓ Preguntas de investigación")
    for c, t in cfg.PREGUNTAS_INVESTIGACION:
        st.markdown(f"- **{c}.** {t}")

    st.markdown("---")
    st.caption(cfg.FOOTER)

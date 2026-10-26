# -*- coding: utf-8 -*-
"""Pestaña 5 · 🤖 Grid2Op POR CIUDAD con aprendizaje por refuerzo — OE-2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import grid2op_rl as rl
from core.ui_helpers import (page_intro, objetivos_panel, kpi_row, AGENT_LABEL,
                             AGENT_COLORS)


@st.cache_data(show_spinner=False)
def _eval_df():
    ev = rl.load_eval()
    if not ev:
        return None, None
    return pd.DataFrame(ev["rows"]), ev


@st.cache_data(show_spinner=False)
def _city_eval():
    return rl.load_city_eval()


@st.cache_data(show_spinner=False)
def _train_hist():
    h = rl.load_train_history()
    return pd.DataFrame(h) if h else None


def _city_eval_table(cev: dict) -> pd.DataFrame:
    """Tabla resumen de las 16 ciudades desde rl_city_eval.json."""
    rows = []
    for city, d in cev["cities"].items():
        s = d["summary"]
        dn, dq = d["donothing"], d["dqn"]
        n = dq["n"]
        rows.append(dict(
            Ciudad=city,
            **{"Pico MW": s["peak_load_mw"], "Ren %": round(s["ren_share"] * 100),
               "Factor pico": s["peak_factor"]},
            **{"DN vive (p)": dn["steps_mean"], f"DN {n}/{n}": f"{dn['survived']}/{n}"},
            **{"DQN vive (p)": dq["steps_mean"], f"DQN {n}/{n}": f"{dq['survived']}/{n}"},
            **{"Ventaja ×": d["ventaja"], "ρ máx DQN": dq["rho_max"]},
        ))
    df = pd.DataFrame(rows)
    return df.sort_values("Ventaja ×", ascending=False).reset_index(drop=True)


def _city_curves_fig(d: dict, metric: str, title: str, hline: float | None = None):
    fig = go.Figure()
    for ag in ("donothing", "dqn"):
        cur = d["curves"].get(ag)
        if not cur:
            continue
        cc = pd.DataFrame(cur)
        fig.add_trace(go.Scatter(
            x=cc["t"], y=cc[metric], name=AGENT_LABEL.get(ag, ag),
            line=dict(color=AGENT_COLORS.get(ag, "#888"), width=2.4)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="red",
                      annotation_text="límite ρ=1 (sobrecarga)")
    fig.update_layout(title=title, height=320, xaxis_title="paso (5 min)",
                      margin=dict(t=44, b=40))
    return fig


def _render_por_ciudad():
    st.subheader("🌐 Grid2Op por ciudad")
    st.markdown(
        "Cada una de las **16 ciudades** tiene su **propio ambiente Grid2Op**: la misma malla "
        "didáctica `rte_case5` (5 subestaciones, 8 líneas), pero con **crónicas re-formadas desde "
        "la curva de demanda sintética de la ciudad** —su factor de pico, su mezcla renovable y su "
        "crecimiento—. Así el **DQN entrenado** (pesos únicos, transferibles porque la topología es "
        "idéntica) se enfrenta a la *personalidad operativa* de cada ciudad y se compara con "
        "**Do-Nothing** en 5 crónicas-semana por ciudad."
    )
    cev = _city_eval()
    if not cev:
        st.warning(
            "Aún no está la evaluación por ciudad. Ejecuta `python scripts/eval_rl_cities.py` "
            "para generar `results/rl_city_eval.json`. Puedes correr un episodio en vivo abajo."
        )
    else:
        df = _city_eval_table(cev)
        n = cev["n_chronics"]
        dqn_surv_all = int(sum(d["dqn"]["survived"] for d in cev["cities"].values()))
        dn_surv_all = int(sum(d["donothing"]["survived"] for d in cev["cities"].values()))
        total_ep = len(cev["cities"]) * n
        mas_dificil = df.iloc[0]  # mayor ventaja = donde Do-Nothing más falla
        kpi_row([
            ("Ciudades evaluadas", f"{len(cev['cities'])}", None, "Ambientes Grid2Op propios"),
            ("DQN · crónicas sobrevividas", f"{dqn_surv_all}/{total_ep}",
             f"{dqn_surv_all/total_ep*100:.0f} %", "Episodios sin game-over (5 crónicas × ciudad)"),
            ("Do-Nothing · sobrevividas", f"{dn_surv_all}/{total_ep}",
             f"{dn_surv_all/total_ep*100:.0f} %", "Línea base sin aprendizaje"),
            ("Ciudad más exigente", f"{mas_dificil['Ciudad']}",
             f"×{mas_dificil['Ventaja ×']}", "Mayor ventaja del DQN sobre Do-Nothing"),
        ])

        st.markdown("**Tabla comparativa por ciudad** (ordenada por ventaja del DQN; "
                    "`vive (p)` = pasos vivos medios de 600, `n/n` = crónicas completas):")
        st.dataframe(
            df.style.format({"Pico MW": "{:.1f}", "DN vive (p)": "{:.0f}",
                             "DQN vive (p)": "{:.0f}", "Ventaja ×": "{:.2f}",
                             "ρ máx DQN": "{:.2f}", "Factor pico": "{:.2f}"})
              .background_gradient(subset=["Ventaja ×"], cmap="RdYlGn")
              .background_gradient(subset=["DN vive (p)"], cmap="Reds"),
            width="stretch", hide_index=True)

        fig = px.bar(df, x="Ciudad", y=["DN vive (p)", "DQN vive (p)"], barmode="group",
                     color_discrete_sequence=[AGENT_COLORS["donothing"], AGENT_COLORS["dqn"]],
                     title="Supervivencia media por ciudad: Do-Nothing vs DQN (pasos de 600)")
        fig.update_layout(height=360, yaxis_title="pasos vivos", legend_title="")
        st.plotly_chart(fig, width="stretch")

        # ---- detalle de la ciudad seleccionada ----
        st.markdown("---")
        st.markdown("#### 🔎 Detalle de una ciudad")
        cities = list(cev["cities"].keys())
        c1, c2 = st.columns([2, 1])
        city = c1.selectbox("Ciudad", cities, index=cities.index("Bogotá") if "Bogotá" in cities else 0,
                            key="rl_city_pane")
        d = cev["cities"][city]
        s = d["summary"]
        c2.metric("Severidad de la crónica", s["peak_factor"], f"ren {s['ren_share']*100:.0f} %")
        kA, kB, kC, kD = st.columns(4)
        kA.metric("Pico de demanda", f"{s['peak_load_mw']:.1f} MW")
        kB.metric("Do-Nothing", f"{d['donothing']['steps_mean']:.0f} p",
                  f"{d['donothing']['survived']}/{d['donothing']['n']} crónicas")
        kC.metric("DQN (tesis)", f"{d['dqn']['steps_mean']:.0f} p",
                  f"{d['dqn']['survived']}/{d['dqn']['n']} crónicas")
        kD.metric("Ventaja del DQN", f"×{d['ventaja']}",
                  help="Ratio de pasos vivos totales DQN / Do-Nothing")

        colA, colB = st.columns(2)
        colA.plotly_chart(_city_curves_fig(d, "reward_cum", f"{city} · recompensa acumulada"),
                          width="stretch")
        colB.plotly_chart(_city_curves_fig(d, "rho_max", f"{city} · congestión ρ máx.", hline=1.0),
                          width="stretch")
        st.caption(f"Crónica 00 de {city} (de {d['dqn']['n']} variantes-semana). "
                   f"ρ>1 ⇒ línea sobrecargada: el DQN la mantiene bajo control; Do-Nothing colapsa.")

        # ---- episodio en vivo sobre el ambiente propio de la ciudad ----
        with st.expander("▶️ Ejecutar episodio EN VIVO en el ambiente propio de la ciudad"):
            st.caption("Reconstruye el ambiente Grid2Op de la ciudad (crónica derivada de su "
                       "demanda) y corre los agentes al instante. La primera vez tarda ~5 s "
                       "(genera la carpeta del ambiente); luego queda en caché.")
            v1, v2 = st.columns([1, 1])
            agents_live = v1.multiselect("Agentes", ["donothing", "dqn", "heuristica"],
                                         default=["donothing", "dqn"],
                                         format_func=lambda a: AGENT_LABEL.get(a, a),
                                         key="rl_city_agents")
            max_steps = v2.slider("Pasos máx.", 100, 600, 300, 50, key="rl_city_steps")
            if st.button("▶️ Correr en vivo", type="primary", key="rl_city_run"):
                if rl.load_agent() is None and "dqn" in agents_live:
                    st.error("No hay pesos de DQN entrenados (models/dqn_case5.npz).")
                else:
                    with st.spinner(f"Simulando Grid2Op en el ambiente de {city}…"):
                        rows = rl.live_eval_city(city, tuple(agents_live), max_steps=max_steps)
                    lv = pd.DataFrame([{
                        "Agente": AGENT_LABEL.get(r["agent"], r["agent"]),
                        "Pasos vivos": r["steps_alive"], "Recompensa": round(r["reward_total"]),
                        "ρ máx": r["rho_max"], "Sobrecargas": r["overload_steps"],
                        "Resultado": "✅ sobrevivió" if r["survived_all"] else "❌ game-over",
                    } for r in rows])
                    st.dataframe(lv, width="stretch", hide_index=True)
                    for r in rows:
                        if r.get("curve"):
                            cc = pd.DataFrame(r["curve"])
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=cc["t"], y=cc["reward_cum"], name="Recompensa",
                                                     line=dict(color=AGENT_COLORS.get(r["agent"], "#888"))))
                            fig.add_trace(go.Scatter(x=cc["t"], y=cc["rho_max"] * 1000, name="ρ máx ×1000",
                                                     yaxis="y2", line=dict(color="#EF476F", dash="dot")))
                            fig.add_hline(y=1000, line_dash="dash", line_color="red", opacity=0.4)
                            fig.update_layout(height=300, yaxis2=dict(overlaying="y", side="right",
                                                                      showgrid=False),
                                              title=f"{AGENT_LABEL.get(r['agent'],r['agent'])} · {city} (en vivo)")
                            st.plotly_chart(fig, width="stretch")


def _render_global(cev_surv_pct: float | None):
    """Evaluación global sobre las 20 crónicas estándar (evidencia agregada)."""
    with st.expander("🎯 Evaluación global · 20 crónicas estándar (evidencia agregada OE-2)"):
        df, ev = _eval_df()
        if df is None:
            st.info("Sin evaluación global precomputada (results/rl_eval.json).")
            return
        summ = df.groupby("agent").agg(
            episodios=("steps_alive", "size"),
            pasos_vivos_prom=("steps_alive", "mean"),
            recompensa_total=("reward_total", "sum"),
            sobrevive_todo=("survived_all", "sum"),
            rho_max_prom=("rho_max", "mean"),
        ).reindex(["donothing", "heuristica", "dqn"]).dropna(how="all")
        summ.index = [AGENT_LABEL.get(i, i) for i in summ.index]
        dqn_alive = float(df[df.agent == "dqn"]["steps_alive"].mean())
        dn_alive = float(df[df.agent == "donothing"]["steps_alive"].mean())
        max_steps = ev.get("max_steps", 800)
        kpi_row([
            ("DQN · pasos vivos prom.", f"{dqn_alive:,.0f} / {max_steps}", None),
            ("Do-Nothing · pasos vivos", f"{dn_alive:,.0f}", None),
            ("Ventaja DQN vs Do-Nothing", f"{dqn_alive/max(dn_alive,1):.1f}×", None),
            ("Crónicas completas (DQN)",
             f"{int(summ.loc[AGENT_LABEL['dqn'],'sobrevive_todo']) if AGENT_LABEL['dqn'] in summ.index else 0}/{ev.get('n_chronics',20)}",
             None),
        ])
        tA, tB = st.columns(2)
        with tA:
            fig = px.bar(summ.reset_index().rename(columns={"index": "agente"}),
                         x="agente", y="pasos_vivos_prom", color="agente",
                         color_discrete_map={AGENT_LABEL[k]: v for k, v in AGENT_COLORS.items()},
                         title="Supervivencia media por agente", text_auto=".0f")
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, width="stretch")
        with tB:
            fig = px.bar(summ.reset_index().rename(columns={"index": "agente"}),
                         x="agente", y="recompensa_total", color="agente",
                         color_discrete_map={AGENT_LABEL[k]: v for k, v in AGENT_COLORS.items()},
                         title="Recompensa acumulada", text_auto=".3s")
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, width="stretch")
        st.dataframe(summ.round(1), width="stretch")


def _render_policy_and_training():
    st.markdown("### 🔎 ¿Qué decide el agente? (inspección de política)")
    cities = sd.list_cities()
    city = st.selectbox("Ciudad para inspeccionar", cities,
                        index=cities.index("Bogotá") if "Bogotá" in cities else 0,
                        key="rl_policy_city")
    if st.button("Inspeccionar Q-values en estado de congestión"):
        insp = rl.inspect_policy(city)
        if insp is None:
            st.error("No hay DQN cargado para inspeccionar.")
        else:
            st.write(f"Estado crítico en crónica `{insp['chronic']}`, paso {insp['step']}, "
                     f"ρ máx = **{insp['rho_max']}** (>1 ⇒ línea sobrecargada).")
            st.bar_chart(pd.Series(insp["rho"], index=[f"L{i}" for i in range(len(insp["rho"]))]),
                         height=200)
            q = pd.DataFrame(insp["top"])
            q.columns = ["Acción candidata", "Q-value", "Legal"]
            st.dataframe(q, width="stretch", hide_index=True)
            st.success(f"🟢 Mejor acción aprendida: **{q.iloc[0]['Acción candidata']}** "
                       f"(Q={q.iloc[0]['Q-value']:.1f}). Trazabilidad de la decisión del agente "
                       f"(actividad 6 · explicabilidad).")

    hist = _train_hist()
    if hist is not None:
        with st.expander("📉 Curva de entrenamiento del DQN (reproducible con scripts/train_dqn.py)"):
            hA, hB = st.columns(2)
            with hA:
                fig = px.line(hist, x="episode", y="steps_alive", markers=True,
                              title="Pasos vivos por episodio de entrenamiento")
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")
            with hB:
                fig = px.line(hist, x="episode", y="epsilon", markers=True,
                              title="Decaimiento de ε (exploración)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")
            st.caption("DQN en NumPy: MLP 60→64→64→25, Adam, replay 20 000, target network, "
                       "ε-greedy 1.0→0.05, máscara de acciones legales. Arquitectura coherente "
                       "con el 'modelo híbrido + refuerzo' del OE-2.")


def render():
    page_intro(
        "grid2op",
        "🤖 Grid2Op por ciudad · operación con aprendizaje por refuerzo · OE-2",
        "Modelo híbrido de operación: un **DQN implementado en NumPy** (arquitectura propia, "
        "reproducible, pesos únicos transferibles) aprende a operar la red de **Grid2Op**. Aquí se "
        "evalúa **en las 16 ciudades**, cada una con su propio ambiente cuyas crónicas se derivan de "
        "su demanda sintética. Se compara con **Do-Nothing** y la heurística **PowerLineSwitch**.",
    )

    _render_por_ciudad()
    st.markdown("---")

    cev = _city_eval()
    surv_pct = None
    if cev:
        tot = len(cev["cities"]) * cev["n_chronics"]
        surv_pct = sum(d["dqn"]["survived"] for d in cev["cities"].values()) / tot * 100
    _render_global(surv_pct)
    _render_policy_and_training()

    # evidencia para el panel de objetivos
    if cev:
        tot = len(cev["cities"]) * cev["n_chronics"]
        surv = sum(d["dqn"]["survived"] for d in cev["cities"].values())
        avg_vent = float(np.mean([d["ventaja"] for d in cev["cities"].values()]))
        ciudades_100 = sum(1 for d in cev["cities"].values() if d["dqn"]["survived"] == d["dqn"]["n"])
    else:
        surv, tot, avg_vent, ciudades_100 = 0, 1, 0.0, 0

    objetivos_panel("grid2op", evidencias={
        "OE-2": [
            ("DQN propio entrenado y reproducible (NumPy, sin cajas negras)", 100),
            (f"Ambiente Grid2Op dedicado para las {len(cev['cities']) if cev else 16} ciudades "
             f"(crónicas desde su demanda)", 100 if cev else 60),
            (f"DQN sobrevive {surv}/{tot} crónicas por ciudad vs Do-Nothing (ventaja media ×{avg_vent:.1f})",
             min(100, surv / max(tot, 1) * 100)),
            (f"{ciudades_100}/{len(cev['cities']) if cev else 16} ciudades con supervivencia 100 %",
             min(100, ciudades_100 / max(len(cev['cities']) if cev else 16, 1) * 100)),
            ("Política inspeccionable (Q-values por acción correctiva)", 100),
            ("Falta: migrar el agente a la red PyPSA específica de cada ciudad (fase siguiente OE-2)", None),
        ],
    }, preguntas=["PI-2", "PI-4", "PI-6"])

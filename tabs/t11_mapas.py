# -*- coding: utf-8 -*-
"""Pestaña 11 · 🗺️ Mapas de Red (grid2poster): transmisión, centrales y solar.

Subpestañas:
  🔌 Red de transmisión   → mapa OSM base + cobertura por tensión + Grid2Op + crecimiento
  🏭 Centrales eléctricas → mapa con --show-plants (OSM power=plant) + mix y ranking
  ☀️ Glow solar           → mapa con EarthPV / OSM+GPPD bajo la red + desglose por fuente
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import synthetic_data as sd
from core import grid2op_rl as rl
from core import population_scenarios as ps
from core.ui_helpers import page_intro, objetivos_panel, kpi_row, AGENT_LABEL, AGENT_COLORS

MAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "city_maps")

TIER_ORDER = ["<60 kV", "60–150 kV", "150–300 kV", ">300 kV"]
TIER_COLORS = {"<60 kV": "#90E0EF", "60–150 kV": "#118AB2", "150–300 kV": "#073B4C",
               ">300 kV": "#EF476F"}
SOURCE_ES = {"solar": "☀️ Solar", "wind": "🌬️ Eólica", "hydro": "💧 Hidro",
             "nuclear": "☢️ Nuclear", "coal": "⛏️ Carbón", "gas": "🔥 Gas",
             "oil": "🛢️ Petróleo", "biomass": "🌿 Biomasa", "waste": "♻️ Residuos",
             "other": "❓ Otra/sin dato"}
SOURCE_COLORS = {"solar": "#FFB703", "wind": "#90E0EF", "hydro": "#118AB2",
                 "nuclear": "#9D0208", "coal": "#6C757D", "gas": "#F4A261",
                 "oil": "#8D6A9F", "biomass": "#06D6A0", "waste": "#A3B18A",
                 "other": "#ADB5BD"}
KIND_COLOR = {"earthpv": "#FFB703", "osm+gppd": "#06D6A0", "sin_datos": "#ADB5BD"}
KIND_LABEL = {"earthpv": "EarthPV (atlas)", "osm+gppd": "OSM + WRI GPPD",
              "sin_datos": "sin datos abiertos"}
EARTHPV_CITIES = {"Casablanca": ("Marruecos", "Bronze"),
                  "Johannesburgo": ("Sudáfrica", "Bronze"),
                  "Hanói": ("Vietnam", "Bronze")}


def _slug(value: str) -> str:
    """Réplica de grid2poster.common.slugify (NFKD + ASCII + no-alfanum → '_')."""
    n = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    n = re.sub(r"[^a-zA-Z0-9]+", "_", n).strip("_").lower()
    return n or "poster"


def _map_path(city: str, variant: str = "base") -> str:
    if variant == "base":
        return os.path.join(MAPS_DIR, f"{_slug(city)}.png")
    return os.path.join(MAPS_DIR, variant, f"{_slug(city)}.png")


@st.cache_data(show_spinner=False)
def _load_json(name: str) -> dict:
    p = os.path.join(MAPS_DIR, name)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _coverage(city: str) -> dict | None:
    return _load_json("coverage.json").get(city)


@st.cache_data(show_spinner=False)
def _city_eval():
    return rl.load_city_eval()


@st.cache_data(show_spinner=False)
def _traj(city, scen, plan):
    return ps.trajectory(city, scen, (), plan)


@st.cache_data(show_spinner=False)
def _concl(city, scen, plan, year):
    return ps.city_conclusion(city, scen, (), plan, year)


def _show_map(city: str, variant: str, caption: str) -> bool:
    p = _map_path(city, variant)
    if os.path.exists(p):
        st.image(p, caption=caption, width="stretch")
        return True
    st.info(f"⏳ El mapa *{variant}* de {city} aún no está renderizado (lote en curso o "
            "pendiente). Ejecuta `python scripts/render_city_maps_layers.py` para generarlo.")
    return False


def _hbar(df: pd.DataFrame, x: str, y: str, color_map: dict | None, title: str,
          highlight: str | None = None, unit: str = "") -> go.Figure:
    fig = go.Figure()
    colors = []
    for v in df[y]:
        if highlight is not None:
            colors.append("#EF476F" if v == highlight else "#B8C4D0")
        elif color_map:
            colors.append(color_map.get(v, "#ADB5BD"))
        else:
            colors.append("#118AB2")
    fig.add_trace(go.Bar(x=df[x], y=df[y], orientation="h", marker_color=colors,
                         text=[f"{v:,.0f}{unit}" for v in df[x]], textposition="outside"))
    fig.update_layout(title=title, height=max(260, 26 * len(df) + 120),
                      margin=dict(t=44, b=30, l=10, r=40), yaxis=dict(autorange="reversed"),
                      xaxis_title=unit.strip() or None)
    return fig


# --------------------------------------------------------------------------- 🔌
def _sub_red(city: str, cities: list[str], c: dict, cev: dict, scen: str, plan: str, year: int):
    cov = _coverage(city)
    mL, mR = st.columns([1.25, 1])
    with mL:
        _show_map(city, "base", f"Red de transmisión real de {city} (OSM vía grid2poster).")
        if cov:
            st.caption(
                f"Cobertura OSM: **{cov.get('total_km', 0):,.0f} km** en "
                f"{cov.get('n_lines', 0)} tramos · " +
                " · ".join(f"{k}: {v:,.0f} km" for k, v in cov.get("tiers", {}).items()))
    with mR:
        st.markdown(f"#### {city} · {c.get('country','')}")
        kpi_row([
            ("Población (2026)", f"{c['pop_m']:.2f} M"),
            ("Factor de pico", f"{c['peak_factor']:.2f}"),
            ("Renovables (mix)", f"{sd.renewable_share(c['mix'])*100:.0f} %"),
            ("Data centers", f"{c['dc_pipeline_mw']:.0f} MW"),
        ], cols_per_row=2)
        if cev and city in cev["cities"]:
            d = cev["cities"][city]
            dn, dq = d["donothing"], d["dqn"]
            st.markdown("**Grid2Op en el ambiente propio de la ciudad** (5 crónicas-semana):")
            kpi_row([
                ("DQN (tesis)", f"{dq['steps_mean']:.0f} p", f"{dq['survived']}/{dq['n']} ✅"),
                ("Do-Nothing", f"{dn['steps_mean']:.0f} p", f"{dn['survived']}/{dn['n']}"),
                ("Ventaja DQN", f"×{d['ventaja']}"),
                ("ρ máx DQN", f"{dq['rho_max']:.2f}"),
            ], cols_per_row=2)
            fig = go.Figure()
            for ag in ("donothing", "dqn"):
                cur = d["curves"].get(ag)
                if cur:
                    cc = pd.DataFrame(cur)
                    fig.add_trace(go.Scatter(x=cc["t"], y=cc["rho_max"],
                                             name=AGENT_LABEL.get(ag, ag),
                                             line=dict(color=AGENT_COLORS.get(ag, "#888"), width=2)))
            fig.add_hline(y=1.0, line_dash="dash", line_color="red")
            fig.update_layout(title=f"{city} · congestión ρ máx (Grid2Op)", height=260,
                              margin=dict(t=40, b=30), xaxis_title="paso (5 min)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Sin evaluación Grid2Op por ciudad todavía (results/rl_city_eval.json).")

    # --- detalle de cobertura por tensión ---
    all_cov = _load_json("coverage.json")
    if cov and cov.get("tiers"):
        dA, dB = st.columns(2)
        tiers = cov["tiers"]
        tdf = pd.DataFrame(
            {"Nivel": [t for t in TIER_ORDER if t in tiers],
             "km": [tiers[t] for t in TIER_ORDER if t in tiers]})
        tdf["% del total"] = (tdf["km"] / max(cov.get("total_km", 1), 1e-9) * 100).round(1)
        with dA:
            st.markdown(f"**Detalle por nivel de tensión — {city}**")
            st.dataframe(tdf, hide_index=True, width="stretch")
            fig = go.Figure(go.Bar(
                x=tdf["km"], y=tdf["Nivel"], orientation="h",
                marker_color=[TIER_COLORS.get(t, "#118AB2") for t in tdf["Nivel"]],
                text=[f"{v:,.0f} km ({p:.0f} %)" for v, p in zip(tdf["km"], tdf["% del total"])],
                textposition="outside"))
            fig.update_layout(title="km mapeados por nivel", height=240,
                              margin=dict(t=40, b=20, l=10, r=40),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        with dB:
            if all_cov:
                cdf = pd.DataFrame({"ciudad": list(all_cov.keys()),
                                    "km": [v.get("total_km", 0) for v in all_cov.values()]}
                                   ).sort_values("km", ascending=False).reset_index(drop=True)
                rank = int(cdf.index[cdf.ciudad == city][0]) + 1 if (cdf.ciudad == city).any() else None
                st.plotly_chart(_hbar(cdf, "km", "ciudad", None,
                                      "Cobertura OSM de transmisión · 16 ciudades (km)",
                                      highlight=city), width="stretch")
                if rank:
                    st.caption(f"{city} ocupa el puesto **{rank}/16** en km de transmisión "
                               "mapeados en OSM (la cobertura depende del trabajo de la "
                               "comunidad MapYourGrid en cada región).")
    else:
        st.info("Sin estadísticas de cobertura (assets/city_maps/coverage.json). Ejecuta "
                "`python scripts/postprocess_coverage.py`.")

    # --- crecimiento poblacional anclado ---
    st.markdown("---")
    st.markdown(f"#### 📈 Conclusión de crecimiento para {city} ({scen} · {plan} · {year})")
    st.info(_concl(city, scen, plan, year))
    tr = _traj(city, scen, plan)
    gA, gB = st.columns(2)
    with gA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.index, y=tr["demanda_TWh"], name="Demanda",
                                 line=dict(color="#EF476F", width=3)))
        fig.add_trace(go.Scatter(x=tr.index, y=tr["oferta_TWh"], name="Oferta (plan)",
                                 fill="tozeroy", fillcolor="rgba(6,214,161,0.15)",
                                 line=dict(color="#06D6A0", width=2)))
        fig.update_layout(title=f"{city} · demanda vs oferta (TWh)", height=300,
                          margin=dict(t=40), xaxis_title="año")
        st.plotly_chart(fig, width="stretch")
    with gB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.index, y=tr["margen_pct"], name="Margen capacidad",
                                 line=dict(color="#118AB2", width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="#9D0208")
        fig.add_hline(y=15, line_dash="dot", line_color="#FFB703")
        fig.update_layout(title=f"{city} · margen de capacidad (%)", height=300,
                          margin=dict(t=40), xaxis_title="año")
        st.plotly_chart(fig, width="stretch")

    with st.expander("🗺️ Galería: mapa base de las 16 ciudades"):
        cols = st.columns(4)
        for i, cty in enumerate(cities):
            with cols[i % 4]:
                p = _map_path(cty, "base")
                if os.path.exists(p):
                    st.image(p, caption=cty, width="stretch")
                else:
                    st.caption(f"⏳ {cty} (pendiente de render)")


# --------------------------------------------------------------------------- 🏭
def _sub_plantas(city: str, cities: list[str]):
    stats = _load_json("plants_stats.json")
    has_map = _show_map(city, "plantas",
                        f"{city} · centrales eléctricas de OSM (power=plant): tamaño = capacidad, "
                        "color = fuente (grid2poster --show-plants).")
    s = stats.get(city)
    if not s:
        st.info("⏳ Estadísticas de centrales pendientes (assets/city_maps/plants_stats.json). "
                "Ejecuta `python scripts/postprocess_layers.py` al terminar el lote.")
        return

    kpi_row([
        ("Centrales mapeadas (OSM)", f"{s['n']}"),
        ("Capacidad total conocida", f"{s['total_mw']:,.0f} MW"),
        ("Con etiqueta de potencia", f"{s['n_known']}/{s['n']}"),
        ("Fuentes distintas", f"{len(s['by_source'])}"),
    ], cols_per_row=4)

    if s["n"] == 0:
        st.warning(f"OSM no tiene centrales `power=plant` mapeadas dentro del límite de {city}: "
                   "el mapa muestra solo la red. Es una limitación real de la cobertura "
                   "comunitaria (supuesto S10.5), no un error del renderizado.")
        return

    cA, cB = st.columns([1, 1.2])
    bs = s["by_source"]
    order = [k for k in SOURCE_ES if k in bs]
    mix = pd.DataFrame({"fuente": [SOURCE_ES.get(k, k) for k in order],
                        "MW": [bs[k]["mw"] for k in order],
                        "n": [bs[k]["n"] for k in order]})
    with cA:
        fig = go.Figure(go.Bar(
            x=mix["MW"], y=mix["fuente"], orientation="h",
            marker_color=[SOURCE_COLORS.get(k, "#ADB5BD") for k in order],
            text=[f"{v:,.0f} MW · {n}" for v, n in zip(mix["MW"], mix["n"])],
            textposition="outside"))
        fig.update_layout(title=f"{city} · capacidad por fuente (MW)", height=300,
                          margin=dict(t=40, b=20, l=10, r=60), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with cB:
        st.markdown(f"**Top 10 centrales de {city} en OSM**")
        top = pd.DataFrame(s["top"])
        top["Fuente"] = top["source"].map(lambda k: SOURCE_ES.get(k, k))
        top = top.rename(columns={"name": "Central", "mw": "MW"})[["Central", "Fuente", "MW"]]
        st.dataframe(top, hide_index=True, width="stretch")

    with st.expander("🌍 Comparación: capacidad mapeada de centrales en las 16 ciudades"):
        rows = []
        for cty in cities:
            v = stats.get(cty, {})
            for k, d in (v.get("by_source") or {}).items():
                rows.append({"ciudad": cty, "fuente": SOURCE_ES.get(k, k), "MW": d["mw"]})
        if rows:
            rdf = pd.DataFrame(rows)
            piv = rdf.pivot_table(index="ciudad", columns="fuente", values="MW",
                                  aggfunc="sum").fillna(0)
            piv = piv.reindex(cities).fillna(0)
            fig = go.Figure()
            for colr in piv.columns:
                key = next((k for k, v in SOURCE_ES.items() if v == colr), "other")
                fig.add_trace(go.Bar(y=piv.index, x=piv[colr], name=colr, orientation="h",
                                     marker_color=SOURCE_COLORS.get(key, "#ADB5BD")))
            fig.update_layout(barmode="stack", title="MW de centrales por fuente y ciudad (OSM)",
                              height=560, margin=dict(t=44, b=30), yaxis=dict(autorange="reversed"),
                              xaxis_title="MW")
            st.plotly_chart(fig, width="stretch")
            st.caption("La cobertura de `power=plant` en OSM varía mucho por país; los MW "
                       "mostrados son los **mapeados por la comunidad**, no el parque real.")

    with st.expander("🗺️ Galería: centrales en las 16 ciudades"):
        cols = st.columns(4)
        for i, cty in enumerate(cities):
            with cols[i % 4]:
                p = _map_path(cty, "plantas")
                if os.path.exists(p):
                    st.image(p, caption=cty, width="stretch")
                else:
                    st.caption(f"⏳ {cty} (pendiente)")
    _ = has_map


# --------------------------------------------------------------------------- ☀️
def _sub_solar(city: str, cities: list[str]):
    stats = _load_json("solar_stats.json")
    _show_map(city, "solar",
              f"{city} · glow de capacidad solar bajo la red (grid2poster --earthpv / --pv-layer).")
    s = stats.get(city)
    if not s:
        st.info("⏳ Estadísticas solares pendientes (assets/city_maps/solar_stats.json). "
                "Ejecuta `python scripts/postprocess_layers.py` al terminar el lote.")
        return

    kind = s.get("kind", "sin_datos")
    src_label = KIND_LABEL.get(kind, kind)
    kpis = [("Capacidad solar en la ciudad", f"{s.get('mwp_ciudad', 0):,.1f} MWp"),
            ("Fuente de la capa", src_label),
            ("Features solares usadas", f"{s.get('n_ciudad', 0):,}")]
    if kind == "earthpv":
        kpis.append(("Atlas EarthPV", f"{s.get('pais')} · tier {s.get('tier')}"))
    kpi_row(kpis, cols_per_row=4)

    if kind == "earthpv":
        nat = s.get("mwp_nacional") or 0
        ciu = s.get("mwp_ciudad", 0)
        cA, cB = st.columns([1, 1])
        with cA:
            fig = go.Figure(go.Bar(
                x=["Atlas nacional (MWp)", f"En {city} (MWp)"], y=[nat, ciu],
                marker_color=["#FFB703", "#EF476F"],
                text=[f"{nat:,.0f}", f"{ciu:,.1f}"], textposition="outside"))
            fig.update_layout(title=f"EarthPV {s.get('pais')} · mejor estimación (mwp_best)",
                              height=280, margin=dict(t=44, b=30))
            st.plotly_chart(fig, width="stretch")
        with cB:
            st.markdown(
                f"**EarthPV** detecta paneles fotovoltaicos en imágenes Sentinel-2 (10 m) y "
                f"publica la capacidad por celda de 0.1° con rango de confianza. El atlas de "
                f"**{s.get('pais')}** (tier *{s.get('tier')}*) estima "
                f"**{nat:,.0f} MWp** nacionales, de los cuales **{ciu:,.1f} MWp** caen dentro "
                f"del límite de {city} (±5 km). Es dato *alfa*: en calibración continua con "
                "la comunidad OSM.")
    elif kind == "osm+gppd":
        desg = s.get("desglose", {})
        ndesg = s.get("n_desglose", {})
        cA, cB = st.columns([1, 1])
        with cA:
            labels = [("Generadores OSM", desg.get("osm", 0.0), ndesg.get("osm", 0)),
                      ("Plantas WRI GPPD", desg.get("gppd", 0.0), ndesg.get("gppd", 0))]
            fig = go.Figure(go.Bar(
                x=[l[0] for l in labels], y=[l[1] for l in labels],
                marker_color=["#06D6A0", "#118AB2"],
                text=[f"{v:,.1f} MWp · {n}" for _, v, n in labels], textposition="outside"))
            fig.update_layout(title=f"{city} · MWp solares por fuente abierta", height=280,
                              margin=dict(t=44, b=30))
            st.plotly_chart(fig, width="stretch")
        with cB:
            st.markdown(
                f"{s.get('pais') or city} no tiene atlas EarthPV publicado. La capa combina "
                f"**{ndesg.get('osm', 0)} generadores solares de OSM** "
                f"({desg.get('osm', 0):,.1f} MWp; sin etiqueta de potencia se asumen 20 kWp, "
                f"S10.7) con **{ndesg.get('gppd', 0)} plantas solares de WRI GPPD** "
                f"({desg.get('gppd', 0):,.1f} MW, foto ≈2018-2020). Posible solapamiento "
                "OSM↔GPPD: el glow es ilustrativo (S10.8).")
    else:
        st.warning(f"No se encontró geodato solar abierto para {city} en ninguna de las tres "
                   "fuentes (EarthPV, generadores OSM, WRI GPPD): el mapa se renderiza sin "
                   "glow. Disponibilidad honesta de datos, no fallo (S10.9).")

    with st.expander("🌍 Comparación solar de las 16 ciudades (MWp dentro del límite)"):
        rows = [{"ciudad": cty, "MWp": stats[cty].get("mwp_ciudad", 0),
                 "fuente": KIND_LABEL.get(stats[cty].get("kind"), stats[cty].get("kind"))}
                for cty in cities if cty in stats]
        if rows:
            df = pd.DataFrame(rows).sort_values("MWp", ascending=False).reset_index(drop=True)
            fig = go.Figure(go.Bar(
                x=df["MWp"], y=df["ciudad"], orientation="h",
                marker_color=[KIND_COLOR.get(k, "#ADB5BD") for k in
                              [next((kk for kk, vv in KIND_LABEL.items()
                                    if vv == f), "sin_datos") for f in df["fuente"]]],
                text=[f"{v:,.1f}" for v in df["MWp"]], textposition="outside"))
            fig.update_layout(title="Capacidad solar georreferenciada por ciudad (MWp)",
                              height=560, margin=dict(t=44, b=30, l=10, r=50),
                              yaxis=dict(autorange="reversed"), xaxis_title="MWp")
            st.plotly_chart(fig, width="stretch")
            st.caption("🟡 EarthPV (atlas alfa por celdas 0.1°) · 🟢 OSM + WRI GPPD · "
                       "⚪ sin datos abiertos. Fuentes y licencias en el desplegable de abajo.")

    with st.expander("📚 Fuentes de la capa solar y cascada de selección"):
        st.markdown(
            "| Prioridad | Fuente | Cobertura | Licencia / nota |\n"
            "|---|---|---|---|\n"
            "| 1 | **EarthPV** (Open Energy Transition) — atlas nacionales por celdas 0.1° "
            "con `mwp_best` | 13 países (de nuestras ciudades: Marruecos, Sudáfrica, Vietnam) "
            "| Abierta; alfa en calibración |\n"
            "| 2 | **OSM** `power=generator` + `generator:source=solar` | Mundial (densidad "
            "según mapeo local) | © OpenStreetMap, ODbL |\n"
            "| 3 | **WRI Global Power Plant Database** (subconjunto solar: 10 665 plantas, "
            "188 GW, 167 países) | Mundial | CC BY 4.0; foto ≈2018-2020 |\n\n"
            "Casablanca, Johannesburgo y Hanói usan el atlas EarthPV de su país; el resto "
            "combina OSM+GPPD. Los archivos fuente se descargan en tiempo de construcción "
            "(ver README); el dashboard solo lee PNG y JSON.")

    with st.expander("🗺️ Galería: glow solar en las 16 ciudades"):
        cols = st.columns(4)
        for i, cty in enumerate(cities):
            with cols[i % 4]:
                p = _map_path(cty, "solar")
                if os.path.exists(p):
                    st.image(p, caption=cty, width="stretch")
                else:
                    st.caption(f"⏳ {cty} (pendiente)")


# --------------------------------------------------------------------- render
def render():
    page_intro(
        "mapas",
        "🗺️ Mapas de Red · grid2poster: transmisión, centrales y glow solar por ciudad",
        "El **mapa real de cada ciudad** se renderiza con **grid2poster** (Open Energy "
        "Transition): límite urbano vía Nominatim + datos **reales de OpenStreetMap** vía "
        "Overpass (GeoPandas + OSMnx + Matplotlib). Tres subpestañas: **🔌 Red de "
        "transmisión** (`power=line`, con cobertura por nivel de tensión, Grid2Op y "
        "crecimiento anclados), **🏭 Centrales eléctricas** (`--show-plants`: `power=plant` "
        "dimensionado por capacidad y coloreado por fuente) y **☀️ Glow solar** "
        "(`--earthpv`/`--pv-layer`: atlas **EarthPV** donde existe —Marruecos, Sudáfrica, "
        "Vietnam— y **OSM + WRI GPPD** como respaldo mundial). Crédito: © OpenStreetMap "
        "(ODbL), MapYourGrid, Open Energy Transition, EarthPV, WRI GPPD (CC BY 4.0).",
    )

    cities = sd.list_cities()
    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    city = c1.selectbox("Ciudad", cities, index=cities.index("Bogotá") if "Bogotá" in cities else 0,
                        key="map_city")
    scen = c2.selectbox("Escenario demográfico", list(ps.SCENARIOS.keys()),
                        index=list(ps.SCENARIOS.keys()).index("Tendencia ONU"), key="map_scen")
    plan = c3.selectbox("Plan de respuesta", list(ps.PLANES.keys()),
                        index=list(ps.PLANES.keys()).index("Robusto-IA (tesis)"), key="map_plan")
    year = st.select_slider("Año horizonte", options=[2030, 2035, 2040, 2045], value=2035,
                            key="map_year")

    cev = _city_eval()
    c = sd.get_city(city)
    cov = _coverage(city)
    pst = _load_json("plants_stats.json").get(city, {})
    sst = _load_json("solar_stats.json").get(city, {})

    # KPIs resumen de las tres capas
    dq = ((cev or {}).get("cities", {}).get(city, {}) or {}).get("dqn", {})
    dn = ((cev or {}).get("cities", {}).get(city, {}) or {}).get("donothing", {})
    vent = (round(dq.get("steps_mean", 0) / dn["steps_mean"], 2)
            if dn.get("steps_mean") else None)
    kpi_row([
        ("Red OSM", f"{cov.get('total_km', 0):,.0f} km" if cov else "—",
         f"{cov.get('n_lines', 0)} tramos" if cov else None),
        ("Centrales OSM", f"{pst.get('n', '—')}" if pst else "⏳",
         f"{pst.get('total_mw', 0):,.0f} MW" if pst else None),
        ("Solar capa", f"{sst.get('mwp_ciudad', 0):,.1f} MWp" if sst else "⏳",
         KIND_LABEL.get(sst.get("kind"), None) if sst else None),
        ("Ventaja DQN", f"×{vent}" if vent else "—"),
    ], cols_per_row=4)

    sub1, sub2, sub3 = st.tabs([
        "🔌 Red de transmisión", "🏭 Centrales eléctricas (OSM)", "☀️ Glow solar (EarthPV)"])
    with sub1:
        _sub_red(city, cities, c, cev, scen, plan, year)
    with sub2:
        _sub_plantas(city, cities)
    with sub3:
        _sub_solar(city, cities)

    objetivos_panel("mapas", evidencias={
        "OE-1": [("Mapa real OSM por ciudad en 3 capas: transmisión (km por tensión), "
                  "centrales (MW por fuente) y solar (EarthPV/OSM/GPPD)", 100)],
        "OE-2": [("Episodio Grid2Op del ambiente propio de la ciudad anclado a su mapa", 100)],
        "OE-3": [("Conclusión de adecuación de capacidad por ciudad bajo escenarios", 100)],
        "OE-4": [("Crecimiento + circunstancias visualizados sobre el territorio real", 90)],
    }, preguntas=["PI-1", "PI-4", "PI-6"])

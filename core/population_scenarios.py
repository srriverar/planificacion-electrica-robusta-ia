"""Escenarios de crecimiento poblacional y circunstancias para las 16 ciudades.

Dos ejes independientes que se cruzan para producir conclusiones:
  • **Escenario demográfico** (`SCENARIOS`): cómo evoluciona la población
    (estancamiento, tendencia ONU, acelerado, boom tecnológico).
  • **Circunstancias** (`CIRCUNSTANCIAS`): eventos/choques activables
    (ola migratoria, olas de calor crónicas, boom de data centers,
     electrificación agresiva, recesión, sequía renovable).
  • **Plan de respuesta** (`PLANES`): cómo reacciona el sistema (BAU, Transición,
     Robusto-IA de la tesis) — determina la oferta/capacidad construida.

Con todo ello se proyecta población → demanda → balance oferta/demanda → margen de
capacidad → CO₂ y se derivan **conclusiones por ciudad** y globales. Es un modelo
anual rápido (capacidad × factor de capacidad) coherente con las reglas de expansión
de PyPSA; el despacho horario fino vive en las pestañas ⚡/🛡️.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core.pypsa_planning as pp
import core.synthetic_data as sd
from core.sustainability import RENS, city_capacity_factors

BASE_YEAR = pp.BASE_YEAR          # 2026
HORIZON = tuple(range(2026, 2046))  # 2026 → 2045
# Tope de saturación de la expansión: ninguna tecnología crece más de SAT× su
# capacidad base (las reglas de expansión de PyPSA están calibradas a 2026-2035;
# sin tope, el crecimiento compuesto diverge de forma irreal en horizontes largos).
SAT_MULT = 4.0

# ---------------------------------------------------------------------------
# Ejes de escenario
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Estancamiento": dict(
        growth_mult=0.2, g_cap=0.02, label="Bajo · envejecimiento", color="#8D99AE",
        desc="Crecimiento poblacional casi nulo (≈20 % de la tendencia): natalidad baja, "
             "envejecimiento y posible despoblación. Presión de demanda contenida."),
    "Tendencia ONU": dict(
        growth_mult=1.0, g_cap=0.055, label="Base · proyección media", color="#118AB2",
        desc="Escenario central: la población sigue su tasa histórica (proxy de "
             "proyecciones ONU/Hábitat). Referencia para comparar el resto."),
    "Acelerado": dict(
        growth_mult=1.5, g_cap=0.06, label="Alto · urbanización rápida", color="#EF476F",
        desc="Urbanización y migración interna aceleradas (≈1,5× la tendencia, tope 6 %/año). "
             "Fuerte presión sobre generación, red y vivienda."),
    "Boom tecnológico": dict(
        growth_mult=1.25, g_cap=0.05, dc_mult=2.6, ev_mult=2.0, label="Alto · hubs de IA y VE",
        color="#FFB703",
        desc="Crecimiento alto + auge de hiperdatacenters (2,6×) y electrificación del "
             "transporte (2×): la demanda crece más rápido que la población."),
}

CIRCUNSTANCIAS = {
    "Ola migratoria": dict(
        label="🚶 Ola migratoria", short="migración +0,8 %/año desde 2028",
        desc="Flujo migratorio sostenido que añade ≈0,8 % anual de población a partir de 2028.",
        efecto="poblacion"),
    "Olas de calor crónicas": dict(
        label="🌡️ Olas de calor crónicas", short="refrigeración +2 %/año",
        desc="Calentamiento sostenido: la demanda por refrigeración sube ≈2 % anual y eleva el pico.",
        efecto="demanda_calor"),
    "Boom de data centers": dict(
        label="🖥️ Boom de data centers", short="carga IA ×2",
        desc="Duplica la expansión de hiperdatacenters (responde a la alerta ProBogotá 2024).",
        efecto="dc"),
    "Electrificación agresiva": dict(
        label="🔌 Electrificación agresiva (VE+bombas)", short="consumo/cápita +1,2 %/año",
        desc="Vehículos eléctricos y bombas de calor elevan el consumo per cápita ≈1,2 % anual.",
        efecto="kwh_pc"),
    "Recesión / desindustrialización": dict(
        label="📉 Recesión", short="demanda −0,4 %/año",
        desc="Choque económico que retrae población y demanda industrial ≈0,4 % anual.",
        efecto="recesion"),
    "Sequía renovable": dict(
        label="🏜️ Sequía renovable", short="hidro/solar/eólica −15 %",
        desc="Sequía y menor recurso recortan ≈15 % la generación renovable variable.",
        efecto="sequia"),
}

PLANES = {
    "BAU tendencial": "BAU tendencial",
    "Transición moderada": "Transición moderada",
    "Robusto-IA (tesis)": "Robusto-IA (tesis)",
}

_CF_CACHE: dict[str, dict] = {}
_BASECAP_CACHE: dict[str, dict] = {}


def _cf(city: str) -> dict:
    if city not in _CF_CACHE:
        _CF_CACHE[city] = city_capacity_factors(city)
    return _CF_CACHE[city]


def _basecap(city: str) -> dict:
    if city not in _BASECAP_CACHE:
        _BASECAP_CACHE[city] = pp.base_capacities(city)
    return _BASECAP_CACHE[city]


def _capped_caps(city: str, year: int, plan: str) -> dict:
    """Capacidades del plan con tope de saturación SAT× respecto a la base."""
    caps = pp.capacities_year(city, year, PLANES[plan])
    base = _basecap(city)
    out = {}
    for tech, cap in caps.items():
        if tech.startswith("_"):
            out[tech] = cap
            continue
        b = base.get(tech, 0.0)
        out[tech] = cap if b <= 0 else min(cap, b * SAT_MULT)
    return out


# ---------------------------------------------------------------------------
# Proyecciones
# ---------------------------------------------------------------------------
def population_m(city: str, year: int, scen: str, circs) -> float:
    """Población (millones) en el año bajo escenario y circunstancias."""
    c = sd.get_city(city)
    k = year - BASE_YEAR
    s = SCENARIOS[scen]
    # tasa efectiva con tope de plausibilidad (evita explosiones demográficas irreales)
    g = min(c["growth_pct"] * s["growth_mult"] / 100.0, s.get("g_cap", 0.06))
    pop = c["pop_m"] * (1 + g) ** k
    if "Ola migratoria" in circs:
        pop *= (1 + 0.005 * max(0, k - 2))
    if "Recesión / desindustrialización" in circs:
        pop *= (1 - 0.004 * max(0, k))
    return pop


def demand_twh(city: str, year: int, scen: str, circs, plan: str) -> dict:
    """Demanda anual (TWh) desglosada, con circunstancias."""
    c = sd.get_city(city)
    k = year - BASE_YEAR
    s = SCENARIOS[scen]
    pop = population_m(city, year, scen, circs)

    # consumo per cápita base + electrificación agresiva
    kwh_pc = c["kwh_pc"]
    if "Electrificación agresiva" in circs:
        kwh_pc *= (1 + 0.012 * k)

    # ahorro por eficiencia del plan de respuesta elegido
    ahorro = pp.capacities_year(city, year, PLANES[plan]).get("_ahorro", 0.0)

    base_twh = pop * kwh_pc / 1000.0
    # calor crónico eleva la demanda de energía
    if "Olas de calor crónicas" in circs:
        base_twh *= (1 + 0.02 * max(0, k - 1))
    # recesión retrae la demanda industrial
    if "Recesión / desindustrialización" in circs:
        base_twh *= (1 - 0.006 * max(0, k))
    base_twh *= (1 - ahorro)

    # data centers (crecen 18 %/año sobre el pipeline) × multiplicadores
    dc_mult = s.get("dc_mult", 1.0) * (2.0 if "Boom de data centers" in circs else 1.0)
    dc_twh = c["dc_pipeline_mw"] * 0.00876 * ((1.18 ** k) - 1) * 0.6 * dc_mult

    return dict(base_twh=base_twh, dc_twh=dc_twh, total_twh=base_twh + dc_twh,
                pop_m=pop, ahorro=ahorro)


def supply_twh(city: str, year: int, plan: str, circs) -> dict:
    """Oferta anual (TWh), CO₂ (Mt) y capacidad efectiva (MW) según el plan."""
    caps = _capped_caps(city, year, plan)
    cf = _cf(city)
    sequia = 0.85 if "Sequía renovable" in circs else 1.0

    total_twh, co2_t, cap_mw_eff, ren_twh = 0.0, 0.0, 0.0, 0.0
    for tech, cap in caps.items():
        if tech.startswith("_") or tech == "bateria" or cap <= 0:
            continue
        f = cf.get(tech, pp.TECH[tech]["cf"] or 0.3)
        if tech in ("hidro", "solar", "eolica", "biomasa") and "Sequía renovable" in circs:
            f *= sequia
        g_twh = cap * f * 8760.0 / 1e6
        total_twh += g_twh
        cap_mw_eff += cap * f
        co2_t += g_twh * 1e6 * pp.TECH[tech]["co2"]
        if tech in RENS:
            ren_twh += g_twh
    return dict(supply_twh=total_twh, co2_Mt=co2_t / 1e6, cap_mw_eff=cap_mw_eff,
                ren_twh=ren_twh, ren_pct=100 * ren_twh / max(total_twh, 1e-9))


def margin_pct(city: str, year: int, scen: str, circs, plan: str) -> float:
    """Margen de capacidad (%) = (capacidad efectiva − pico de demanda)/pico."""
    c = sd.get_city(city)
    dem = demand_twh(city, year, scen, circs, plan)["total_twh"]
    avg_mw = dem * 1e6 / 8760.0
    peak_mw = avg_mw * c["peak_factor"]
    if "Olas de calor crónicas" in circs:
        peak_mw *= 1.05
    sup = supply_twh(city, year, plan, circs)["cap_mw_eff"]
    return (sup - peak_mw) / max(peak_mw, 1e-9) * 100.0


def capacity_need_gw(city: str, year: int, scen: str, circs, plan: str,
                     target_margin: float = 15.0) -> float:
    """GW de capacidad **efectiva** adicional para alcanzar `target_margin` en el año.
    Es la brecha de adecuación de pico (no la de energía)."""
    c = sd.get_city(city)
    dem = demand_twh(city, year, scen, circs, plan)["total_twh"]
    peak_mw = dem * 1e6 / 8760.0 * c["peak_factor"]
    if "Olas de calor crónicas" in circs:
        peak_mw *= 1.05
    sup = supply_twh(city, year, plan, circs)["cap_mw_eff"]
    need_mw = peak_mw * (1 + target_margin / 100.0) - sup
    return max(0.0, need_mw) / 1000.0


def capacity_gap_year(city: str, scen: str, circs, plan: str,
                      years=HORIZON) -> int | None:
    """Primer año con margen de capacidad NEGATIVO (déficit de adecuación real)."""
    for y in years:
        m = margin_pct(city, y, scen, circs, plan)
        if m is not None and m < 0.0:
            return y
    return None


def stress_year(city: str, scen: str, circs, plan: str, years=HORIZON) -> int | None:
    """Primer año con margen < 15 % (zona de estrés aunque aún sin déficit)."""
    for y in years:
        m = margin_pct(city, y, scen, circs, plan)
        if m is not None and m < 15.0:
            return y
    return None


def trajectory(city: str, scen: str = "Tendencia ONU", circs=(), plan: str = "Robusto-IA (tesis)",
               years=HORIZON) -> pd.DataFrame:
    """Serie anual completa para una ciudad bajo escenario + circunstancias + plan."""
    rows = []
    for y in years:
        dem = demand_twh(city, y, scen, circs, plan)
        sup = supply_twh(city, y, plan, circs)
        marg = margin_pct(city, y, scen, circs, plan)
        deficit = max(dem["total_twh"] - sup["supply_twh"], 0.0)
        # CO₂ efectivo: oferta + déficit cubierto con gas (proxy marginal)
        co2_eff = sup["co2_Mt"] + deficit * 1e6 * pp.TECH["gas"]["co2"] / 1e6
        rows.append(dict(
            year=y, pop_m=dem["pop_m"], demanda_TWh=dem["total_twh"],
            demanda_dc_TWh=dem["dc_twh"], oferta_TWh=sup["supply_twh"],
            deficit_GWh=deficit * 1000.0, margen_pct=marg,
            renovables_pct=sup["ren_pct"], co2_Mt=co2_eff,
            capacidad_efectiva_MW=sup["cap_mw_eff"],
            pico_MW=dem["total_twh"] * 1e6 / 8760.0 * sd.get_city(city)["peak_factor"],
        ))
    return pd.DataFrame(rows).set_index("year")


def risk_level(marg: float) -> tuple[str, str]:
    """Clasificación de riesgo por margen de capacidad → (etiqueta, color)."""
    if marg < 0:
        return "Crítico · déficit", "#9D0208"
    if marg < 15:
        return "Riesgo alto", "#EF476F"
    if marg < 30:
        return "Vigilar", "#FFB703"
    return "Estable", "#06D6A0"


def assess_all(scen: str = "Tendencia ONU", circs=(), plan: str = "Robusto-IA (tesis)",
               year: int = 2035) -> pd.DataFrame:
    """Resumen comparativo de las 16 ciudades en un año horizonte."""
    rows = []
    for city in sd.list_cities():
        c = sd.get_city(city)
        base = trajectory(city, scen, circs, plan, years=(BASE_YEAR, year))
        d0, d1 = base.loc[BASE_YEAR], base.loc[year]
        pop0 = population_m(city, BASE_YEAR, scen, circs)
        marg = float(d1["margen_pct"])
        riesgo, color = risk_level(marg)
        yrs = tuple(range(BASE_YEAR, year + 1))
        gap_y = capacity_gap_year(city, scen, circs, plan, years=yrs)
        extra_gw = capacity_need_gw(city, year, scen, circs, plan)
        rows.append(dict(
            Ciudad=city, Pais=c.get("country", ""),
            Pob_2026_M=round(pop0, 2), Pob_M=round(float(d1["pop_m"]), 2),
            dPob_pct=round((float(d1["pop_m"]) / pop0 - 1) * 100, 1),
            Demanda_TWh=round(float(d1["demanda_TWh"]), 1),
            dDemanda_pct=round((float(d1["demanda_TWh"]) / float(d0["demanda_TWh"]) - 1) * 100, 1),
            Margen_pct=round(marg, 1),
            Deficit_GWh=round(float(d1["deficit_GWh"]), 1),
            Nueva_cap_GW=round(extra_gw, 2),
            Ren_pct=round(float(d1["renovables_pct"]), 1),
            CO2_Mt=round(float(d1["co2_Mt"]), 2),
            Anio_deficit=gap_y if gap_y else "—",
            Riesgo=riesgo, _color=color,
        ))
    df = pd.DataFrame(rows)
    return df.sort_values("Margen_pct").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Conclusiones (generación de lenguaje)
# ---------------------------------------------------------------------------
def city_conclusion(city: str, scen: str = "Tendencia ONU", circs=(),
                    plan: str = "Robusto-IA (tesis)", year: int = 2035) -> str:
    c = sd.get_city(city)
    tr = trajectory(city, scen, circs, plan, years=(BASE_YEAR, year))
    d0, d1 = tr.loc[BASE_YEAR], tr.loc[year]
    pop0 = population_m(city, BASE_YEAR, scen, circs)
    dpob = (float(d1["pop_m"]) / pop0 - 1) * 100
    ddem = (float(d1["demanda_TWh"]) / float(d0["demanda_TWh"]) - 1) * 100
    marg = float(d1["margen_pct"])
    riesgo, _ = risk_level(marg)
    gap_y = capacity_gap_year(city, scen, circs, plan, years=tuple(range(BASE_YEAR, year + 1)))
    extra_gw = capacity_need_gw(city, year, scen, circs, plan)
    partes = [f"**{city}** ({c.get('country','')}) — riesgo **{riesgo}** en {year}."]
    partes.append(
        f"Población {pop0:.2f}→{float(d1['pop_m']):.2f} M ({dpob:+.0f} %) y demanda "
        f"{float(d0['demanda_TWh']):.1f}→{float(d1['demanda_TWh']):.1f} TWh ({ddem:+.0f} %)."
    )
    if gap_y:
        partes.append(
            f"⚠️ Déficit de adecuación de pico desde **{gap_y}**; se requieren ≈"
            f"**{extra_gw:.2f} GW** efectivos adicionales (o gestión de demanda) para {year}."
        )
    elif marg < 15:
        partes.append(
            f"Margen ajustado ({marg:.0f} %) en {year}: ≈**{extra_gw:.2f} GW** de holgura "
            f"recomendable para no entrar en estrés."
        )
    else:
        partes.append(
            f"El plan mantiene margen ≥ 15 % hasta {year} ({marg:.0f} %); sin déficit proyectado."
        )
    # matices por circunstancias
    if "Boom de data centers" in circs or scen == "Boom tecnológico":
        dc = float(d1["demanda_dc_TWh"])
        partes.append(f"Los data centers aportan {dc:.1f} TWh ({dc/float(d1['demanda_TWh'])*100:.0f} % de la demanda).")
    if "Sequía renovable" in circs:
        partes.append("La sequía recorta la renovable y eleva el CO₂ efectivo: priorizar almacenamiento y respaldo firme.")
    if "Olas de calor crónicas" in circs:
        partes.append("El calor crónico sube el pico ≈5 %: refuerzar red y respuesta de demanda.")
    # recomendación
    if marg < 0:
        rec = "Acción urgente: expansión acelerada + eficiencia + integración regional."
    elif marg < 15:
        rec = "Acelerar renovables+almacenamiento y gestión de la demanda antes del año crítico."
    elif marg < 30:
        rec = "Monitorear y anticipar inversiones; el plan Robusto-IA da holgura moderada."
    else:
        rec = "Escenario cómodo: sostener la transición y exportar excedentes limpios."
    partes.append(f"💡 {rec}")
    return " ".join(partes)


def global_conclusions(scen: str = "Tendencia ONU", circs=(),
                       plan: str = "Robusto-IA (tesis)", year: int = 2035) -> list[str]:
    """Conclusiones agregadas para las 16 ciudades."""
    df = assess_all(scen, circs, plan, year)
    n = len(df)
    criticos = df[df["Margen_pct"] < 0]["Ciudad"].tolist()
    riesgo_alto = df[(df["Margen_pct"] >= 0) & (df["Margen_pct"] < 15)]["Ciudad"].tolist()
    estables = df[df["Margen_pct"] >= 30]["Ciudad"].tolist()
    mas_gap = df.sort_values("Margen_pct").head(3)
    nueva_cap = df["Nueva_cap_GW"].sum()
    dco2 = df["CO2_Mt"].sum()
    pop_total = df["Pob_M"].sum()

    s = SCENARIOS[scen]
    circ_txt = "sin circunstancias adicionales" if not circs else "con " + ", ".join(circs).lower()
    out = [
        f"🌐 **Escenario «{s['label']}» · plan {plan} · horizonte {year}** ({circ_txt}). "
        f"Población agregada ≈{pop_total:.0f} M en las 16 ciudades.",
        f"🟢 **{len(estables)}/{n} ciudades estables** (margen ≥ 30 %): "
        f"{', '.join(estables[:6])}{'…' if len(estables) > 6 else ''}.",
    ]
    if criticos:
        out.append(f"🔴 **Déficit proyectado en {len(criticos)} ciudad(es):** {', '.join(criticos)}. "
                   f"Requieren expansión inmediata o gestión de demanda agresiva.")
    if riesgo_alto:
        out.append(f"🟠 **Riesgo alto (margen < 15 %) en {len(riesgo_alto)} ciudad(es):** "
                   f"{', '.join(riesgo_alto)}.")
    out.append(
        f"⚡ Necesidad agregada de **≈{nueva_cap:.0f} GW** de capacidad nueva para {year}. "
        f"Las 3 más tensionadas: {', '.join(f'{r.Ciudad} ({r.Margen_pct:.0f} %)' for r in mas_gap.itertuples())}."
    )
    out.append(
        f"🌱 Emisiones agregadas del suministro ≈**{dco2:.1f} Mt CO₂** en {year}. "
        f"El plan Robusto-IA y las circunstancias renovables determinan la trayectoria."
    )
    # recomendación de política
    if len(criticos) + len(riesgo_alto) >= n * 0.4:
        out.append("🧭 **Conclusión de política:** bajo este escenario la planificación debe volverse "
                   "*proactiva y robusta* (OE-2/OE-3): adelantar renovables+almacenamiento, respuesta de "
                   "demanda e integración regional antes de que los márgenes colapsen.")
    else:
        out.append("🧭 **Conclusión de política:** el sistema aguanta el escenario, pero la robustez exige "
                   "monitoreo continuo (OE-3) y margen para choques (resiliencia, pestaña 🛡️).")
    return out

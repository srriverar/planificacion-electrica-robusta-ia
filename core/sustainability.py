# -*- coding: utf-8 -*-
"""
Modelo anual rápido de sostenibilidad (OE-4).

Balance capacidad×factor-de-capacidad por ciudad/año/escenario (sin LP) para
trayectorias 2026–2035 multi-ciudad: CO2, participación renovable, OPEX,
inversión, riesgo de déficit e índice de eficiencia. Consistente con las reglas
de expansión de `pypsa_planning` (mismas capacidades), pero ~1000× más rápido,
apto para comparar las 16 ciudades en el dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import synthetic_data as sd
from . import pypsa_planning as pp
from .config import METAS

RENS = ("hidro", "solar", "eolica", "geotermica", "biomasa")


def city_capacity_factors(city: str) -> dict:
    """Factores de capacidad por tecnología (calibrados con perfiles sintéticos)."""
    prof = pp.seasonal_profiles(city)
    cf = {}
    for tech, t in pp.TECH.items():
        if t["cf"] is None:
            continue
        cf[tech] = t["cf"]
    cf["solar"] = float(np.clip(np.mean([p["solar_pu"].mean() for p in prof.values()]) * 1.05, 0.08, 0.32))
    cf["eolica"] = float(np.clip(np.mean([p["wind_pu"].mean() for p in prof.values()]) * 1.6, 0.15, 0.45))
    # hidro estacional promedio con factor de planta
    cf["hidro"] = float(np.clip(0.45 * np.mean([1.15, 1.0, 0.85, 1.0]), 0.3, 0.55))
    return cf


def annual_balance(city: str, year: int, scenario: str, co2_price: float = 30.0,
                   gas_shock: float = 0.0) -> dict:
    caps = pp.capacities_year(city, year, scenario, co2_price, gas_shock)
    cf = city_capacity_factors(city)
    ahorro = caps.get("_ahorro", 0.0)
    dem_twh = pp.demand_year(city, year, ahorro)
    dem_mwh = dem_twh * 1e6

    gen_mwh, co2_t, opex = {}, 0.0, 0.0
    for tech, cap in caps.items():
        if tech.startswith("_") or tech == "bateria" or cap <= 0:
            continue
        t = pp.TECH[tech]
        g = cap * cf.get(tech, t["cf"] or 0.3) * 8760.0  # MW × cf × 8760 h = MWh/año
        gen_mwh[tech] = g
        co2_t += g * t["co2"]
        mc = t["mc"] + t["co2"] * caps.get("_co2_price", 30.0)
        if tech == "gas":
            mc *= (1 + caps.get("_gas_shock", 0.0))
        opex += g * mc
    supply_mwh = sum(gen_mwh.values())
    deficit_mwh = max(dem_mwh - supply_mwh, 0.0)
    surplus_mwh = max(supply_mwh - dem_mwh, 0.0)
    ren_mwh = sum(g for k, g in gen_mwh.items() if k in RENS)
    c = sd.get_city(city)

    # Inversión del año (delta de capacidades × capex)
    inv = 0.0
    if year > pp.BASE_YEAR:
        prev = pp.capacities_year(city, year - 1, scenario, co2_price, gas_shock)
        for tech, cap in caps.items():
            if tech.startswith("_") or tech == "bateria":
                continue
            d = cap - prev.get(tech, 0.0)
            if d > 0:
                inv += d * 1000 * pp.TECH[tech]["capex"]
        db = caps.get("bateria", 0.0) - prev.get("bateria", 0.0)
        if db > 0:
            inv += db * 1000 * pp.TECH["bateria"]["capex"] * 4  # energía 4 h

    # Índice de eficiencia global (sintético, documentado en supuestos):
    # combina pérdidas de red, ahorro acumulado por eficiencia/respuesta de
    # demanda y factor de utilización de la infraestructura.
    eff_base = 1.0 - c["losses_pct"] / 100.0
    util = min(1.0, 0.75 + 0.25 * supply_mwh / max(dem_mwh, 1.0))
    eficiencia_raw = 100 * eff_base * (1 + ahorro * 6) * util
    eff_idx = min(100.0, eficiencia_raw)
    return dict(
        year=year, scenario=scenario, city=city,
        demanda_TWh=dem_twh, oferta_TWh=supply_mwh / 1e6,
        deficit_GWh=deficit_mwh / 1000, surplus_GWh=surplus_mwh / 1000,
        renovables_pct=100 * ren_mwh / max(supply_mwh, 1),
        co2_Mt=co2_t / 1e6, co2_g_kwh=co2_t * 1e3 / max(supply_mwh, 1),
        opex_MUSD=opex / 1e6, inversion_MUSD=inv / 1e6,
        eficiencia_idx=eff_idx, eficiencia_raw=eficiencia_raw,
        generacion_por_tech={k: v / 1000 for k, v in gen_mwh.items()},  # GWh
    )


def trajectory(city: str, scenario: str, years: tuple = tuple(range(2026, 2036)),
               co2_price: float = 30.0, gas_shock: float = 0.0) -> pd.DataFrame:
    rows = [annual_balance(city, y, scenario, co2_price, gas_shock) for y in years]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "generacion_por_tech"} for r in rows])
    return df.set_index("year")


def metas_progress(city: str, co2_price: float = 30.0, gas_shock: float = 0.0) -> pd.DataFrame:
    """Avance del modelo Robusto-IA frente a las metas cuantitativas de la
    propuesta (actividades SR4): CO2 −20 %, renovables +25 %, OPEX −15 %,
    eficiencia +30 % (horizonte 2026→2031, 5 años)."""
    y0, y1 = 2026, 2031
    bau0 = annual_balance(city, y0, "BAU tendencial", co2_price, gas_shock)
    rob1 = annual_balance(city, y1, "Robusto-IA (tesis)", co2_price, gas_shock)
    bau1 = annual_balance(city, y1, "BAU tendencial", co2_price, gas_shock)
    d_co2 = (rob1["co2_Mt"] - bau1["co2_Mt"]) / max(bau1["co2_Mt"], 1e-9) * 100
    d_ren = rob1["renovables_pct"] - bau1["renovables_pct"]
    d_opex = (rob1["opex_MUSD"] - bau1["opex_MUSD"]) / max(bau1["opex_MUSD"], 1e-9) * 100
    d_eff = (rob1["eficiencia_raw"] - bau1["eficiencia_raw"]) / max(bau1["eficiencia_raw"], 1e-9) * 100
    rows = [
        dict(meta="Reducción de CO2 en generación", objetivo=f"−{METAS['co2_menos_pct']:.0f} % vs BAU ({y1})",
             logrado=f"{d_co2:+.1f} %", cumple=d_co2 <= -METAS["co2_menos_pct"] * 0.999,
             avance_pct=min(100, max(0, -d_co2 / METAS["co2_menos_pct"] * 100))),
        dict(meta="Aumento de participación renovable", objetivo=f"+{METAS['renovables_extra_pct']:.0f} pp vs BAU ({y1})",
             logrado=f"{d_ren:+.1f} pp", cumple=d_ren >= METAS["renovables_extra_pct"] * 0.999,
             avance_pct=min(100, max(0, d_ren / METAS["renovables_extra_pct"] * 100))),
        dict(meta="Reducción de costos operativos", objetivo=f"−{METAS['opex_menos_pct']:.0f} % vs BAU ({y1})",
             logrado=f"{d_opex:+.1f} %", cumple=d_opex <= -METAS["opex_menos_pct"] * 0.999,
             avance_pct=min(100, max(0, -d_opex / METAS["opex_menos_pct"] * 100))),
        dict(meta="Eficiencia energética global", objetivo=f"+{METAS['eficiencia_global_pct']:.0f} % vs BAU ({y1})",
             logrado=f"{d_eff:+.1f} %", cumple=d_eff >= METAS["eficiencia_global_pct"] * 0.999,
             avance_pct=min(100, max(0, d_eff / METAS["eficiencia_global_pct"] * 100))),
    ]
    return pd.DataFrame(rows)


def radar_indices(city: str, year: int = 2030) -> dict:
    """Índices 0–100 de las tres dimensiones de variables de la propuesta."""
    c = sd.get_city(city)
    rob = annual_balance(city, year, "Robusto-IA (tesis)")
    # Demanda: presión de crecimiento (crecimiento + VE + data centers)
    avg_mw = c["pop_m"] * c["kwh_pc"] / 8760 * 1000  # MW promedio
    dc_share = min(30, c["dc_pipeline_mw"] / max(avg_mw, 1) * 100)
    demanda_idx = float(np.clip(25 + c["growth_pct"] * 7 + c["ev_share"] * 90 + dc_share * 1.1, 0, 100))
    # Resiliencia: SAIDI, pérdidas y almacenamiento del plan robusto
    caps = pp.capacities_year(city, year, "Robusto-IA (tesis)")
    peak = caps["_peak_mw"] * (1 + c["growth_pct"] / 100) ** (year - pp.BASE_YEAR)
    firm = sum(v for k, v in caps.items() if k in ("hidro", "geotermica", "nuclear", "gas",
               "carbon", "petroleo", "biomasa", "importaciones") and not k.startswith("_"))
    margen = 100 * (firm + caps.get("bateria", 0) * 0.5) / max(peak, 1) - 100
    bat_pct = 100 * caps.get("bateria", 0) / max(peak, 1)
    resiliencia_idx = float(np.clip(100 - c["saidi_h"] * 1.8 - max(c["losses_pct"] - 5, 0) * 1.5
                                    + min(bat_pct, 12) * 1.5, 0, 100))
    # Sostenibilidad: renovables + intensidad de carbono del plan robusto
    sostenibilidad_idx = float(np.clip(rob["renovables_pct"] * 0.7
                                       + (1 - min(rob["co2_g_kwh"], 600) / 600) * 45, 0, 100))
    return dict(Demanda=demanda_idx, Resiliencia=resiliencia_idx,
                Sostenibilidad=sostenibilidad_idx, margen_reserva_pct=margen)

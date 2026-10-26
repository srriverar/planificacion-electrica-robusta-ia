# -*- coding: utf-8 -*-
"""
Planificación de sistemas eléctricos con PyPSA sobre la red sintética de cada ciudad.

- Red agregada de 7 buses (interconexión 220 kV, centro 220 kV, 5 zonas 110 kV).
- Despacho óptimo (LOPF/DC) sobre 96 snapshots representativos (4 estaciones × 24 h).
- Expansión año a año 2026→2035 con reglas por escenario (BAU, Transición, Robusto-IA).
- Contingencias N-1 con flujo lineal y eventos extremos con racionamiento explícito.

Supuestos principales: flujo DC sin unit-commitment, CAPEX simplificados,
VOLL = 500 USD/MWh, parque inicial coherente con la matriz sintética de la ciudad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import logging

for _lg in ("linopy", "pypsa"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

from . import synthetic_data as sd

VOLL = 500.0           # USD/MWh valor de pérdida de carga
BASE_YEAR = 2026

ZONES = ["Norte", "Sur", "Este", "Oeste", "Industrial"]
ZONE_SHARE = {"Centro": 0.22, "Norte": 0.18, "Sur": 0.18, "Este": 0.16,
              "Oeste": 0.14, "Industrial": 0.12}

# tecnología -> (marginal base USD/MWh, capex USD/kW, intensidad tCO2/MWh, cap factor base)
TECH = {
    "hidro":        dict(mc=4.0,  capex=2000, co2=0.0,  cf=0.45),
    "geotermica":   dict(mc=7.0,  capex=2800, co2=0.0,  cf=0.80),
    "nuclear":      dict(mc=10.0, capex=6000, co2=0.0,  cf=0.85),
    "biomasa":      dict(mc=26.0, capex=2200, co2=0.12, cf=0.60),
    "gas":          dict(mc=38.0, capex=950,  co2=0.42, cf=0.38),
    "carbon":       dict(mc=30.0, capex=1600, co2=0.95, cf=0.55),
    "petroleo":     dict(mc=78.0, capex=700,  co2=0.65, cf=0.15),
    "solar":        dict(mc=2.0,  capex=650,  co2=0.0,  cf=0.20),
    "eolica":       dict(mc=2.5,  capex=1250, co2=0.0,  cf=0.28),
    "importaciones":dict(mc=70.0, capex=0,    co2=0.45, cf=0.40),
    "bateria":      dict(mc=1.5,  capex=380,  co2=0.0,  cf=None),
}

SEASONS = {"Ene": 1, "Abr": 4, "Jul": 7, "Oct": 10}
SEASON_DAYS = {"Ene": 90, "Abr": 92, "Jul": 92, "Oct": 91}

ESCENARIOS = ["BAU tendencial", "Transición moderada", "Robusto-IA (tesis)"]

# reglas de expansión anual por escenario: {tech: crecimiento anual compuesto}
EXPANSION_RULES = {
    "BAU tendencial":       dict(solar=0.06, eolica=0.03, gas=0.035, carbon=0.01,
                                 bateria=0.05, hidro=0.005, ahorro_demanda=0.0, margen=1.18),
    "Transición moderada":  dict(solar=0.15, eolica=0.12, gas=0.01, carbon=-0.03,
                                 bateria=0.22, hidro=0.005, ahorro_demanda=0.004, margen=1.25),
    "Robusto-IA (tesis)":   dict(solar=0.22, eolica=0.18, gas=0.0, carbon=-0.07,
                                 bateria=0.42, hidro=0.01, ahorro_demanda=0.009, margen=1.35),
}


# ---------------------------------------------------------------------------
# Construcción de snapshots y perfiles representativos
# ---------------------------------------------------------------------------
def representative_snapshots(end_year: int = 2026) -> tuple[pd.DatetimeIndex, pd.Series]:
    snaps, weights = [], []
    for sname, month in SEASONS.items():
        day = pd.Timestamp(year=end_year, month=month, day=15)
        idx = pd.date_range(day, periods=24, freq="h")
        snaps.append(idx)
        weights.append(pd.Series(SEASON_DAYS[sname], index=idx))
    snap_idx = pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in snaps]))
    w = pd.concat(weights)
    w.index = snap_idx
    return snap_idx, w


def seasonal_profiles(city: str, end_year: int = 2026) -> dict:
    """Perfiles horarios por estación (demanda por zona, solar, eólica, hidro, precio)."""
    df = sd.city_hourly(city, years=2, end_year=end_year)
    df = df[df.index.year == end_year - 1]  # último año completo disponible
    out = {}
    for sname, month in SEASONS.items():
        m = df[df.index.month == month]
        by_hour = m.groupby(m.index.hour).mean(numeric_only=True)
        out[sname] = by_hour
    return out


# ---------------------------------------------------------------------------
# Capacidades por escenario y año
# ---------------------------------------------------------------------------
def base_capacities(city: str) -> dict:
    """Parque instalado 2026 (MW) coherente con la matriz sintética y el pico."""
    c = sd.get_city(city)
    base_avg_mw = c["pop_m"] * 1e6 * c["kwh_pc"] / 8760.0 / 1000.0
    peak = base_avg_mw * c["peak_factor"] + c["dc_pipeline_mw"] * 0.6
    caps = {}
    for tech, share in c["mix"].items():
        t = TECH[tech]
        cf = t["cf"]
        if tech == "solar":
            prof = seasonal_profiles(city)
            cf = float(np.mean([p["solar_pu"].mean() for p in prof.values()])) * 1.05
            cf = max(cf, 0.10)
        if tech == "eolica":
            prof = seasonal_profiles(city)
            cf = float(np.mean([p["wind_pu"].mean() for p in prof.values()])) * 1.6
            cf = float(np.clip(cf, 0.15, 0.45))
        caps[tech] = peak * 1.15 * share / cf
    # batería inicial: 2 % del pico (MW, 4 h)
    caps["bateria"] = peak * 0.02
    caps["_peak_mw"] = peak
    return caps


def capacities_year(city: str, year: int, scenario: str, co2_price: float = 30.0,
                    gas_shock: float = 0.0) -> dict:
    caps = base_capacities(city)
    rules = EXPANSION_RULES[scenario]
    k = year - BASE_YEAR
    out = {}
    for tech, cap in caps.items():
        if tech.startswith("_"):
            out[tech] = cap * (1 + sd.get_city(city)["growth_pct"] / 100) ** k
            continue
        g = rules.get(tech, 0.005)
        out[tech] = cap * (1 + g) ** k
    # ahorro de demanda por eficiencia (Robusto-IA incluye respuesta de demanda)
    out["_ahorro"] = rules["ahorro_demanda"] * k
    out["_margen"] = rules["margen"]
    out["_co2_price"] = co2_price
    out["_gas_shock"] = gas_shock
    return out


def demand_year(city: str, year: int, ahorro: float = 0.0) -> float:
    """Demanda anual TWh del año con tendencia y ahorro acumulado."""
    c = sd.get_city(city)
    base_twh = c["pop_m"] * c["kwh_pc"] / 1000.0
    k = year - BASE_YEAR
    dc_c = c["dc_pipeline_mw"] * 0.00876 * ((1.18 ** k) - 1) * 0.6  # TWh extra data centers
    return base_twh * ((1 + c["growth_pct"] / 100) ** k) * (1 - ahorro) + dc_c


# ---------------------------------------------------------------------------
# Red PyPSA
# ---------------------------------------------------------------------------
def build_network(city: str, year: int = BASE_YEAR, scenario: str = "Robusto-IA (tesis)",
                  co2_price: float = 30.0, gas_shock: float = 0.0,
                  demand_scale: float = 1.0, solar_derate: float = 1.0,
                  wind_derate: float = 1.0, outage_lines: list[int] | None = None,
                  gen_outage: str | None = None) -> pypsa.Network:
    """Construye la red sintética de la ciudad lista para n.optimize()."""
    caps = capacities_year(city, year, scenario, co2_price, gas_shock)
    prof = seasonal_profiles(city)
    c = sd.get_city(city)
    ahorro = caps.get("_ahorro", 0.0)

    n = pypsa.Network()
    n.name = f"{city} {year} · {scenario}"
    snaps, weights = representative_snapshots(year)
    n.set_snapshots(snaps)
    n.snapshot_weightings.loc[:, :] = 1.0
    for col in n.snapshot_weightings.columns:
        n.snapshot_weightings[col] = weights.to_numpy()

    # Buses
    n.add("Bus", "Interconexión", v_nom=220)
    n.add("Bus", "Centro", v_nom=220)
    for z in ZONES:
        n.add("Bus", z, v_nom=110)

    # Carriers
    for tech, t in TECH.items():
        n.add("Carrier", tech, co2_emissions=t["co2"])
    n.add("Carrier", "racionamiento", co2_emissions=0.0)

    # Cargas por zona (perfil estacional horario escalado al año)
    peak = caps["_peak_mw"] * (1 + c["growth_pct"] / 100) ** (year - BASE_YEAR)
    load_series = {}
    for zname, share in ZONE_SHARE.items():
        vals = []
        for sname in SEASONS:
            p = prof[sname]["demand_mw"].to_numpy()
            vals.append(p * share * peak / prof[sname]["demand_mw"].sum() * 24)
        v = np.concatenate(vals) * (1 - ahorro) * demand_scale
        n.add("Load", f"carga_{zname}", bus=zname if zname != "Centro" else "Centro",
              p_set=v)
        load_series[zname] = v

    # Líneas: anillo 220 kV + radiales 110 kV
    hv_ring = [("Interconexión", "Centro", 1800), ("Centro", "Norte", 900),
               ("Norte", "Este", 700), ("Este", "Sur", 700), ("Sur", "Oeste", 700),
               ("Oeste", "Centro", 900), ("Centro", "Industrial", 800),
               ("Norte", "Sur", 500), ("Este", "Oeste", 500)]
    for i, (b0, b1, s_nom) in enumerate(hv_ring):
        x = 0.008 if "Interconexión" in (b0, b1) or "Centro" == b0 or "Centro" == b1 else 0.015
        n.add("Line", f"L{i}_{b0[:3]}-{b1[:3]}", bus0=b0, bus1=b1, x=x, r=x / 8,
              s_nom=s_nom * (1.0 + 0.12 * (year - BASE_YEAR)))
    line_names = list(n.lines.index)
    if outage_lines:
        for li in outage_lines:
            if 0 <= li < len(line_names):
                n.remove("Line", line_names[li])

    # Generadores
    solar_pu = np.concatenate([prof[s]["solar_pu"].to_numpy() for s in SEASONS]) * solar_derate
    wind_pu = np.concatenate([prof[s]["wind_pu"].to_numpy() for s in SEASONS]) * wind_derate
    hydro_season = {"Ene": 1.15, "Abr": 1.0, "Jul": 0.85, "Oct": 1.0}
    hydro_pu = np.concatenate([
        np.full(24, 0.45 * hydro_season[s] * (1.15 if c["lat"] < 0 and s in ("Ene",) else 1.0))
        for s in SEASONS])

    bus_for_tech = {"hidro": "Centro", "geotermica": "Centro", "nuclear": "Centro",
                    "gas": "Norte", "carbon": "Industrial", "petroleo": "Industrial",
                    "biomasa": "Sur", "solar": "Oeste", "eolica": "Este",
                    "importaciones": "Interconexión"}
    gen_names = {}
    for tech, cap in caps.items():
        if tech.startswith("_") or tech == "bateria" or cap <= 1e-6:
            continue
        t = TECH[tech]
        mc = t["mc"] + t["co2"] * caps.get("_co2_price", 30.0)
        if tech == "gas":
            mc *= (1 + caps.get("_gas_shock", 0.0))
        kw = dict(bus=bus_for_tech[tech], carrier=tech, p_nom=round(float(cap), 1),
                  marginal_cost=round(mc, 2))
        if tech == "solar":
            kw["p_max_pu"] = np.clip(solar_pu, 0, 1)
        elif tech == "eolica":
            kw["p_max_pu"] = np.clip(wind_pu, 0, 1)
        elif tech in ("hidro", "geotermica"):
            kw["p_max_pu"] = np.clip(hydro_pu, 0, 1)
        elif tech == "importaciones":
            kw["p_nom"] = round(float(cap * 1.2), 1)
        gname = f"gen_{tech}"
        if gen_outage and tech == gen_outage:
            kw["p_nom"] = 0.0
        n.add("Generator", gname, **kw)
        gen_names[tech] = gname

    # Batería (4 h)
    bat_p = caps.get("bateria", 0.0)
    if bat_p > 1e-6:
        n.add("StorageUnit", "bateria", bus="Centro", carrier="bateria",
              p_nom=round(float(bat_p), 1), max_hours=4, efficiency_store=0.92,
              efficiency_dispatch=0.92, standing_loss=0.0005, marginal_cost=1.5,
              cyclic_state_of_charge=True)

    # Racionamiento en cada zona (ENS explícito)
    for zname in list(ZONE_SHARE.keys()):
        bus = zname
        n.add("Generator", f"racion_{zname}", bus=bus, carrier="racionamiento",
              p_nom=round(float(load_series[zname].max()) + 50, 1),
              marginal_cost=VOLL)
    return n


def solve(n: pypsa.Network, solver: str = "highs") -> str:
    try:
        status, _ = n.optimize(solver_name=solver, solver_options={"output_flag": False})
    except Exception:
        status, _ = n.optimize(solver_name=solver)
    return str(status)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
def dispatch_results(n: pypsa.Network) -> dict:
    """Extrae KPIs y series del despacho resuelto."""
    w = n.snapshot_weightings["objective"].to_numpy()
    gen = n.generators_t.p.to_numpy()
    carriers = n.generators.carrier.to_numpy()
    p_nom = n.generators.p_nom.to_numpy()
    energy = {}
    for i, c in enumerate(n.generators.index):
        energy[c] = float((gen[:, i] * w).sum())
    ens_mwh = sum(v for k, v in energy.items() if k.startswith("racion"))
    total_load = float((n.loads_t.p_set.to_numpy() * w[:, None]).sum())
    ren_techs = ("hidro", "solar", "eolica", "geotermica", "biomasa")
    ren_mwh = sum(v for k, v in energy.items() if k.replace("gen_", "") in ren_techs)
    co2_t = 0.0
    for cname, e in energy.items():
        tech = cname.replace("gen_", "")
        co2_t += e * TECH.get(tech, dict(co2=0.0))["co2"]  # MWh × t/MWh
    bat = n.storage_units_t
    soc = bat.state_of_charge["bateria"] if "bateria" in bat.state_of_charge.columns else None
    line_load = (n.lines_t.p0.abs().max() / n.lines.s_nom * 100)
    prices = n.buses_t.marginal_price

    # Curtailment renovable: (p_disponible − p_despachada) ponderada
    curt = {}
    for tech in ("solar", "eolica"):
        gname = f"gen_{tech}"
        if gname in n.generators.index and gname in n.generators_t.p_max_pu.columns:
            avail = n.generators_t.p_max_pu[gname].to_numpy() * n.generators.p_nom[gname]
            used = n.generators_t.p[gname].to_numpy()
            curt[tech] = float(((avail - used) * w).sum())
        else:
            curt[tech] = 0.0

    return dict(
        energy_by_gen=pd.Series(energy).sort_values(ascending=False),
        ens_mwh=ens_mwh, total_load_mwh=total_load,
        renewable_share_pct=100 * ren_mwh / max(total_load, 1e-9),
        co2_t=co2_t, co2_g_per_kwh=co2_t * 1e3 / max(total_load, 1e-9),
        objective_usd=float(n.objective),
        line_max_loading_pct=float(line_load.max()),
        line_loading=line_load.sort_values(ascending=False),
        soc=soc, prices=prices, gen_p=n.generators_t.p, curtailment=curt,
    )


def run_year(city: str, year: int, scenario: str, co2_price: float = 30.0,
             gas_shock: float = 0.0, **event) -> dict:
    """Resuelve un año/escenario y devuelve KPIs anuales."""
    n = build_network(city, year, scenario, co2_price, gas_shock, **event)
    status = solve(n)
    res = dispatch_results(n)
    res["status"] = status
    res["year"] = year
    res["scenario"] = scenario
    res["city"] = city
    caps = capacities_year(city, year, scenario, co2_price, gas_shock)
    res["capacity_mw"] = {k: v for k, v in caps.items() if not k.startswith("_")}
    res["peak_mw"] = caps["_peak_mw"] * (1 + sd.get_city(city)["growth_pct"] / 100) ** (year - BASE_YEAR)
    firm = sum(v for k, v in caps.items()
               if k in ("hidro", "geotermica", "nuclear", "gas", "carbon", "petroleo",
                        "biomasa", "importaciones") and not k.startswith("_"))
    res["reserve_margin_pct"] = 100 * (firm + caps.get("bateria", 0) * 0.5 +
                                       0.3 * (caps.get("solar", 0) + caps.get("eolica", 0))) / res["peak_mw"] - 100
    res["demand_twh"] = demand_year(city, year, caps.get("_ahorro", 0.0))
    inv = 0.0
    if year > BASE_YEAR:
        prev = capacities_year(city, year - 1, scenario, co2_price, gas_shock)
        for tech, cap in caps.items():
            if tech.startswith("_"):
                continue
            d = cap - prev.get(tech, 0.0)
            if d > 0:
                inv += d * 1000 * TECH[tech]["capex"]
    res["investment_usd"] = inv
    return res


def expansion_trajectory(city: str, years: tuple[int, ...], scenario: str,
                         co2_price: float = 30.0, gas_shock: float = 0.0) -> pd.DataFrame:
    """Trayectoria multi-año (LP por año, cacheable)."""
    rows = []
    for y in years:
        r = run_year(city, y, scenario, co2_price, gas_shock)
        rows.append({
            "year": y, "costo_MUSD": r["objective_usd"] / 1e6,
            "co2_Mt": r["co2_t"] / 1e6, "co2_g_kwh": r["co2_g_per_kwh"],
            "ren_pct": r["renewable_share_pct"], "ENS_MWh": r["ens_mwh"],
            "margen_reserva_pct": r["reserve_margin_pct"],
            "inversion_MUSD": r["investment_usd"] / 1e6,
            "demanda_TWh": r["demand_twh"],
        })
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# N-1 y eventos extremos
# ---------------------------------------------------------------------------
def n1_contingency(city: str, year: int = BASE_YEAR, scenario: str = "Robusto-IA (tesis)",
                   co2_price: float = 30.0) -> pd.DataFrame:
    """Análisis N-1: por cada línea, se reconstruye la red sin ella y se re-despacha
    (redispatch + racionamiento explícito). Viola N-1 si hay ENS > 0 o carga > 100 %."""
    import networkx as nx
    base_n = build_network(city, year, scenario, co2_price)
    line_names = list(base_n.lines.index)
    rows = []
    # caso base
    status = solve(base_n)
    rb = dispatch_results(base_n)
    rows.append(dict(contingencia="— caso base —", red_conectada=True,
                     max_carga_pct=round(rb["line_max_loading_pct"], 1),
                     ENS_MWh=round(rb["ens_mwh"], 1),
                     costo_extra_MUSD=0.0, viola_n1=bool(rb["ens_mwh"] > 1)))
    base_cost = rb["objective_usd"]
    for i, lname in enumerate(line_names):
        nc = build_network(city, year, scenario, co2_price, outage_lines=[i])
        connected = nx.is_connected(nc.graph())
        try:
            st = solve(nc)
            r = dispatch_results(nc)
            mx, ens, cost = r["line_max_loading_pct"], r["ens_mwh"], r["objective_usd"]
            if st != "ok":
                mx, ens, cost = np.nan, np.nan, np.nan
        except Exception:
            mx, ens, cost = np.nan, np.nan, np.nan
        rows.append(dict(contingencia=lname, red_conectada=connected,
                         max_carga_pct=round(float(mx), 1) if mx == mx else np.nan,
                         ENS_MWh=round(float(ens), 1) if ens == ens else np.nan,
                         costo_extra_MUSD=round((cost - base_cost) / 1e6, 2) if cost == cost else np.nan,
                         viola_n1=bool((ens == ens and ens > 1) or (mx == mx and mx > 100) or not connected)))
    return pd.DataFrame(rows)


EVENTS = {
    "Pico de demanda extremo": dict(kind="demand", msg="Ola de calor regional + evento masivo: la demanda sube un X %"),
    "Ola de calor": dict(kind="heat", msg="Derating térmico 8 %, solar −15 %, demanda de refrigeración +X %"),
    "Tormenta / huracán": dict(kind="storm", msg="Salen 2 líneas, eólica −50 %, solar −40 %, demanda +X/3 %"),
    "Falla de generación": dict(kind="genfail", msg="Sale la mayor planta térmica y la demanda sube X/2 %"),
    "Surgimiento data center": dict(kind="dc", msg="Nuevo hiperdata center añade X % de carga plana 24/7"),
}


def apply_event_kwargs(kind: str, severity: float) -> dict:
    x = severity
    if kind == "demand":
        return dict(demand_scale=1 + x)
    if kind == "heat":
        return dict(demand_scale=1 + x, solar_derate=0.85)
    if kind == "storm":
        return dict(demand_scale=1 + x / 3, solar_derate=0.60, wind_derate=0.50,
                    outage_lines=[2, 5])
    if kind == "genfail":
        return dict(demand_scale=1 + x / 2, gen_outage="gas")
    if kind == "dc":
        return dict(demand_scale=1 + x)
    return {}


def event_kpis(city: str, event_name: str, severity: float, year: int = 2030,
               co2_price: float = 30.0) -> pd.DataFrame:
    """Compara modo determinista (BAU) vs robusto-IA ante un evento."""
    kind = EVENTS[event_name]["kind"]
    kw = apply_event_kwargs(kind, severity)
    rows = []
    for mode, scen in [("Determinista tradicional", "BAU tendencial"),
                       ("Robusto-IA (tesis)", "Robusto-IA (tesis)")]:
        r = run_year(city, year, scen, co2_price, **kw)
        rows.append(dict(modo=mode, escenario=scen,
                         ENS_MWh=round(r["ens_mwh"], 0),
                         ENS_pct=round(100 * r["ens_mwh"] / max(r["total_load_mwh"], 1), 3),
                         costo_MUSD=round(r["objective_usd"] / 1e6, 1),
                         co2_g_kwh=round(r["co2_g_per_kwh"], 0),
                         max_carga_linea_pct=round(r["line_max_loading_pct"], 1),
                         renovables_pct=round(r["renewable_share_pct"], 1),
                         margen_reserva_pct=round(r["reserve_margin_pct"], 1)))
    return pd.DataFrame(rows)


def severity_sweep(city: str, event_name: str, year: int = 2030,
                   co2_price: float = 30.0) -> pd.DataFrame:
    sevs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    out = []
    for s in sevs:
        df = event_kpis(city, event_name, s, year, co2_price)
        df["severidad_pct"] = round(s * 100)
        out.append(df)
    return pd.concat(out, ignore_index=True)

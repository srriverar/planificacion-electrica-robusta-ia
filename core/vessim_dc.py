# -*- coding: utf-8 -*-
"""
Co-simulación de data centers con Vessim (OE-4 / PI-1).

Modela la interacción entre la carga computacional de un hiperdatacenter urbano,
PV on-site, baterías y la red pública de la ciudad sintética, con señales de
intensidad de carbono y precio. Políticas comparadas:
  - sin_gestion:      data center tradicional, todo lo importa de la red.
  - bateria_default:  batería + DefaultDispatchPolicy de Vessim.
  - carbon_aware:     política propia que carga la batería cuando la intensidad
                      de carbono es baja y descarga cuando es alta (despacho
                      consciente de carbono), demostrando gestión activa.

Unidades: Vessim trabaja en vatios (W); las trazas de ciudad vienen en MW.
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import pandas as pd
import vessim as vs

from . import synthetic_data as sd

STEP_MIN_DEFAULT = 15
SIM_START = "2026-03-01"

POLITICAS = ["sin_gestion", "bateria_default", "carbon_aware"]
POLITICA_LABEL = {
    "sin_gestion": "Sin gestión (DC tradicional)",
    "bateria_default": "Batería + despacho por defecto",
    "carbon_aware": "Carbon-aware (tesis)",
}


# ---------------------------------------------------------------------------
# Trazas sintéticas
# ---------------------------------------------------------------------------
def _city_window(city: str, days: int, step_min: int) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    df = sd.city_hourly(city, years=1)
    hours = days * 24 + 24  # margen
    df = df.iloc[:hours]
    idx = pd.date_range(df.index[0], periods=hours * 60 // step_min, freq=f"{step_min}min")
    res = df.reindex(df.index.union(idx)).interpolate("time").loc[idx]
    return idx, res


def dc_load_w(city: str, days: int, step_min: int = STEP_MIN_DEFAULT,
              pue: float = 1.30, phase_frac: float = 0.60,
              workload_shift: bool = False) -> pd.Series:
    """Carga total del data center (W): IT (inferencia diurna + entrenamiento
    nocturno) × PUE. `workload_shift` mueve parte del entrenamiento flexible a
    horas de baja intensidad de carbono (computación carbon-aware)."""
    c = sd.get_city(city)
    idx, res = _city_window(city, days, step_min)
    pipeline_w = c["dc_pipeline_mw"] * phase_frac * 1e6
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    dow = idx.dayofweek.to_numpy()
    # inferencia: pico horario comercial; entrenamiento: base plana + pico nocturno
    inference = 0.55 * (0.65 + 0.35 * np.exp(-((hour - 14) ** 2) / 14.0)) * np.where(dow < 5, 1.0, 0.82)
    training = 0.45 * (0.80 + 0.20 * np.exp(-((hour - 2) ** 2) / 10.0))
    rng = sd._rng(city, "dc")
    util = np.clip(inference + training + rng.normal(0, 0.03, len(idx)), 0.35, 0.98)
    it_w = pipeline_w * util
    if workload_shift:
        ci = res["ci_gco2_kwh"].to_numpy()
        thr = np.quantile(ci, 0.35)
        flex = 0.18 * pipeline_w * (training / np.maximum(training.max(), 1e-9))
        it_w = np.where(ci <= thr, it_w + flex, it_w - 0.35 * flex)
        it_w = np.clip(it_w, 0.2 * pipeline_w, 1.05 * pipeline_w)
    return pd.Series(it_w * pue, index=idx, name="dc_load_w")


def solar_w(city: str, days: int, solar_mw: float, step_min: int = STEP_MIN_DEFAULT) -> pd.Series:
    idx, res = _city_window(city, days, step_min)
    return pd.Series(res["solar_pu"].to_numpy() * solar_mw * 1e6, index=idx, name="pv_w")


def grid_signal(city: str, days: int, col: str, step_min: int = STEP_MIN_DEFAULT) -> pd.Series:
    idx, res = _city_window(city, days, step_min)
    return pd.Series(res[col].to_numpy(), index=idx, name=col)


# ---------------------------------------------------------------------------
# Política carbon-aware propia
# ---------------------------------------------------------------------------
class CarbonAwarePolicy(vs.DefaultDispatchPolicy):
    """Carga la batería con red baja en carbono y descarga en horas sucias/caras."""

    def __init__(self, ci_low: float, ci_high: float, charge_max_w: float,
                 price_weight: float = 0.0):
        super().__init__()
        self.ci_low, self.ci_high = ci_low, ci_high
        self.charge_max_w = charge_max_w
        self.price_weight = price_weight

    def apply(self, p_delta, duration, dispatchables, grid_signals=None):
        gs = grid_signals or {}
        ci = float(gs.get("carbon_intensity", (self.ci_low + self.ci_high) / 2))
        price = float(gs.get("price", 0.0))
        score = ci + self.price_weight * price * 2.0  # gCO2/kWh equivalente
        if score < self.ci_low:
            self.charge_power = self.charge_max_w
        elif score > self.ci_high:
            self.charge_power = -self.charge_max_w
        else:
            self.charge_power = 0.0
        super().apply(p_delta, duration, dispatchables, grid_signals)


# ---------------------------------------------------------------------------
# Co-simulación
# ---------------------------------------------------------------------------
def run_dc_microgrid(city: str, days: int = 30, step_min: int = STEP_MIN_DEFAULT,
                     pue: float = 1.30, phase_frac: float = 0.60,
                     solar_frac: float = 0.30, battery_frac: float = 0.50,
                     policy: str = "carbon_aware", workload_shift: bool = False) -> dict:
    """Ejecuta la microred del data center con Vessim y devuelve KPIs + series.

    solar_frac / battery_frac se dimensionan sobre la potencia del pipeline
    activo del data center (MW): PV = solar_frac×pipeline, batería =
    battery_frac×pipeline (MWh, C-rate 0.5).
    """
    c = sd.get_city(city)
    pipeline_mw = c["dc_pipeline_mw"] * phase_frac
    load = dc_load_w(city, days, step_min, pue, phase_frac, workload_shift)
    solar = solar_w(city, days, pipeline_mw * solar_frac, step_min)
    ci_sig = grid_signal(city, days, "ci_gco2_kwh", step_min)
    price_sig = grid_signal(city, days, "price_eur_mwh", step_min)

    step_s = step_min * 60
    env = vs.Environment(sim_start=SIM_START, step_size=step_s)
    actors = [vs.Consumer("it_cooling", signal=vs.Trace(load, anchor=load.index[0])),
              vs.Producer("pv_onsite", signal=vs.Trace(solar, anchor=solar.index[0]))]
    dispatchables: list = []
    bat_mwh = pipeline_mw * battery_frac
    if policy in ("bateria_default", "carbon_aware") and bat_mwh > 0:
        dispatchables.append(vs.SimpleBattery("bess", capacity=bat_mwh * 1e6,
                                              initial_soc=0.5, c_rate=0.5))
    dispatchables.append(vs.GridConnection("red_ciudad"))

    pol_kwargs = {}
    if policy == "carbon_aware":
        # umbral sobre la MISMA escala combinada que usa la política (CI + w·precio)
        price_scale = float(ci_sig.std() / max(price_sig.std(), 1e-9)) * 0.5
        score = ci_sig.to_numpy() + price_scale * price_sig.to_numpy()
        lo, hi = np.quantile(score, [0.30, 0.70])
        pol_kwargs = dict(policy=CarbonAwarePolicy(ci_low=float(lo), ci_high=float(hi),
                                                   charge_max_w=bat_mwh * 0.5 * 1e6,
                                                   price_weight=price_scale))
    env.add_microgrid(name="datacenter", actors=actors, dispatchables=dispatchables,
                      grid_signals={"carbon_intensity": vs.Trace(ci_sig, anchor=ci_sig.index[0]),
                                    "price": vs.Trace(price_sig, anchor=price_sig.index[0])},
                      **pol_kwargs)

    out = tempfile.mkdtemp(prefix="vessim_")
    rec = env.add_controller(vs.Recorder(os.path.join(out, "run")))
    env.run(until=days * 24 * 3600, print_progress=False)
    df = rec.to_df()
    df = df.xs("datacenter", level="microgrid") if "microgrid" in df.index.names else df

    p_grid = df["p_grid"].to_numpy()                      # + exporta, − importa (W)
    p_delta = df["p_delta"].to_numpy()                    # + excedente actores
    hours = step_min / 60.0
    import_w = np.clip(-p_grid, 0, None)                  # W importados
    export_w = np.clip(p_grid, 0, None)                   # W exportados
    import_wh = float((import_w * hours).sum())
    export_wh = float((export_w * hours).sum())
    load_wh = float((-np.clip(p_delta, None, 0) * hours).sum())
    pv_wh = float(np.clip(df["actor_states.pv_onsite.power"].to_numpy(), 0, None).sum() * hours)
    ci = df["grid_signals.carbon_intensity"].to_numpy() if "grid_signals.carbon_intensity" in df.columns else ci_sig.reindex(df.index).to_numpy()
    price = df["grid_signals.price"].to_numpy() if "grid_signals.price" in df.columns else price_sig.reindex(df.index).to_numpy()
    # Energía por paso (kWh) y emisiones (g) → tCO2; costo (MWh × €/MWh) → €
    import_kwh_step = import_w * hours / 1000.0
    emissions_g = float((import_kwh_step * ci).sum())
    import_mwh_step = import_w * hours / 1e6
    cost_eur = float((import_mwh_step * price).sum())
    total_hours = max(len(df) * hours, 1e-9)
    avg_load_mw = load_wh / total_hours / 1e6
    kpis = dict(
        ciudad=city, politica=POLITICA_LABEL[policy],
        carga_dc_mwh=round(load_wh / 1e6, 1),
        importado_red_mwh=round(import_wh / 1e6, 1),
        exportado_red_mwh=round(export_wh / 1e6, 1),
        pv_generado_mwh=round(pv_wh / 1e6, 1),
        autosuficiencia_pct=round(100 * (1 - import_wh / max(load_wh, 1)), 1),
        emisiones_tco2=round(emissions_g / 1e6, 2),
        costo_red_eur=round(cost_eur, 0),
        pico_import_mw=round(float(import_w.max()) / 1e6, 2),
        avg_dc_mw=round(avg_load_mw, 1),
        share_demanda_ciudad_pct=round(100 * avg_load_mw / max(_city_avg_mw(city), 1e-9), 2),
    )
    return dict(kpis=kpis, df=df, config=dict(days=days, step_min=step_min, pue=pue,
               phase_frac=phase_frac, solar_frac=solar_frac, battery_frac=battery_frac,
               policy=policy, workload_shift=workload_shift, pipeline_mw=pipeline_mw))


def _city_avg_mw(city: str) -> float:
    c = sd.get_city(city)
    return c["pop_m"] * 1e6 * c["kwh_pc"] / 8760.0 / 1000.0


def compare_policies(city: str, days: int = 30, step_min: int = STEP_MIN_DEFAULT,
                     pue: float = 1.30, solar_frac: float = 0.30,
                     battery_frac: float = 0.50, workload_shift: bool = False) -> pd.DataFrame:
    """Compara las tres políticas de gestión en la misma configuración."""
    rows = []
    for pol in POLITICAS:
        r = run_dc_microgrid(city, days, step_min, pue, phase_frac=0.60,
                             solar_frac=solar_frac, battery_frac=battery_frac,
                             policy=pol, workload_shift=workload_shift and pol == "carbon_aware")
        rows.append(r["kpis"])
    return pd.DataFrame(rows)

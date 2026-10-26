# -*- coding: utf-8 -*-
"""
Datos sintéticos mundiales para el dashboard de la tesis.

Catálogo de ciudades cuya población actual (2025, ciudad o área metropolitana
según se indique) está entre 6 y 9 millones de habitantes, más generadores
determinísticos (semilla fija) de series horarias de demanda, clima, renovables,
intensidad de carbono y precio.

Supuesto global: TODO es sintético salvo el orden de magnitud de las
poblaciones y las matrices energéticas, calibradas a valores públicos plausibles.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

SEED_BASE = 20261023

# Intensidades de CO2 por tecnología (gCO2/kWh) — valores tipo IPCC
CO2_FACTORS = {
    "hidro": 24, "nuclear": 12, "solar": 45, "eolica": 11, "geotermica": 38,
    "gas": 420, "carbon": 950, "petroleo": 650, "biomasa": 120, "importaciones": 450,
}

# ---------------------------------------------------------------------------
# Catálogo de ciudades (población 6–9 M hab.)
# ---------------------------------------------------------------------------
CITIES: list[dict] = [
    dict(name="Bogotá", country="Colombia", region="América del Sur", role="Capital (caso de estudio)",
         pop_scope="ciudad", lat=4.61, lon=-74.08, pop_m=7.95, kwh_pc=1550, growth_pct=2.4,
         ev_share=0.02, dc_pipeline_mw=140, climate="tropical de altura", temp_mean=14.0, temp_amp_season=1.5,
         temp_amp_diurnal=6.0, heating_elec=0.02, cooling_sens=0.9, sectors=dict(res=0.42, com=0.36, ind=0.22),
         peak_factor=1.45, saidi_h=6.0, losses_pct=10.5, tz=-5,
         mix=dict(hidro=0.62, gas=0.17, carbon=0.04, solar=0.05, eolica=0.03, importaciones=0.09)),
    dict(name="Londres", country="Reino Unido", region="Europa", role="Capital",
         pop_scope="ciudad", lat=51.51, lon=-0.13, pop_m=8.80, kwh_pc=5200, growth_pct=1.5,
         ev_share=0.12, dc_pipeline_mw=380, climate="templado marítimo", temp_mean=11.5, temp_amp_season=7.0,
         temp_amp_diurnal=5.0, heating_elec=0.20, cooling_sens=0.5, sectors=dict(res=0.38, com=0.44, ind=0.18),
         peak_factor=1.60, saidi_h=1.0, losses_pct=7.0, tz=0,
         mix=dict(eolica=0.30, gas=0.31, nuclear=0.15, solar=0.07, biomasa=0.05, importaciones=0.12)),
    dict(name="Nueva York", country="Estados Unidos", region="América del Norte", role="Metrópoli global",
         pop_scope="ciudad", lat=40.71, lon=-74.01, pop_m=8.40, kwh_pc=7000, growth_pct=1.3,
         ev_share=0.06, dc_pipeline_mw=450, climate="continental húmedo", temp_mean=13.0, temp_amp_season=12.0,
         temp_amp_diurnal=7.0, heating_elec=0.30, cooling_sens=1.3, sectors=dict(res=0.36, com=0.46, ind=0.18),
         peak_factor=1.70, saidi_h=2.0, losses_pct=6.5, tz=-5,
         mix=dict(gas=0.44, nuclear=0.26, hidro=0.13, solar=0.07, eolica=0.05, importaciones=0.05)),
    dict(name="Teherán", country="Irán", region="Asia", role="Capital",
         pop_scope="ciudad", lat=35.69, lon=51.39, pop_m=8.80, kwh_pc=3600, growth_pct=2.8,
         ev_share=0.01, dc_pipeline_mw=60, climate="semiárido", temp_mean=17.5, temp_amp_season=11.0,
         temp_amp_diurnal=9.0, heating_elec=0.10, cooling_sens=1.4, sectors=dict(res=0.44, com=0.30, ind=0.26),
         peak_factor=1.65, saidi_h=6.5, losses_pct=12.0, tz=3.5,
         mix=dict(gas=0.78, hidro=0.08, solar=0.06, eolica=0.05, petroleo=0.03)),
    dict(name="Bangkok", country="Tailandia", region="Asia", role="Capital",
         pop_scope="área administrativa", lat=13.76, lon=100.50, pop_m=8.40, kwh_pc=3000, growth_pct=2.6,
         ev_share=0.04, dc_pipeline_mw=180, climate="tropical monzónico", temp_mean=28.5, temp_amp_season=2.5,
         temp_amp_diurnal=6.0, heating_elec=0.0, cooling_sens=1.8, sectors=dict(res=0.38, com=0.40, ind=0.22),
         peak_factor=1.50, saidi_h=4.0, losses_pct=9.0, tz=7,
         mix=dict(gas=0.56, carbon=0.14, solar=0.10, importaciones=0.10, hidro=0.06, eolica=0.04)),
    dict(name="Lima", country="Perú", region="América del Sur", role="Capital",
         pop_scope="metropolitana", lat=-12.05, lon=-77.04, pop_m=8.60, kwh_pc=1600, growth_pct=3.0,
         ev_share=0.02, dc_pipeline_mw=70, climate="desértico costero", temp_mean=19.5, temp_amp_season=4.0,
         temp_amp_diurnal=5.0, heating_elec=0.02, cooling_sens=0.8, sectors=dict(res=0.46, com=0.32, ind=0.22),
         peak_factor=1.45, saidi_h=7.0, losses_pct=10.0, tz=-5,
         mix=dict(gas=0.44, hidro=0.37, eolica=0.10, solar=0.07, petroleo=0.02)),
    dict(name="Santiago", country="Chile", region="América del Sur", role="Capital",
         pop_scope="metropolitana", lat=-33.45, lon=-70.67, pop_m=6.70, kwh_pc=4200, growth_pct=2.5,
         ev_share=0.05, dc_pipeline_mw=110, climate="mediterráneo", temp_mean=15.0, temp_amp_season=9.0,
         temp_amp_diurnal=10.0, heating_elec=0.15, cooling_sens=0.9, sectors=dict(res=0.40, com=0.36, ind=0.24),
         peak_factor=1.55, saidi_h=2.5, losses_pct=7.5, tz=-4,
         mix=dict(hidro=0.24, solar=0.22, eolica=0.19, gas=0.13, carbon=0.09, importaciones=0.13)),
    dict(name="Río de Janeiro", country="Brasil", region="América del Sur", role="Metrópoli costera",
         pop_scope="municipio", lat=-22.91, lon=-43.17, pop_m=6.50, kwh_pc=2400, growth_pct=2.0,
         ev_share=0.03, dc_pipeline_mw=150, climate="tropical sabana", temp_mean=24.0, temp_amp_season=4.5,
         temp_amp_diurnal=6.0, heating_elec=0.02, cooling_sens=1.5, sectors=dict(res=0.43, com=0.35, ind=0.22),
         peak_factor=1.50, saidi_h=6.0, losses_pct=11.0, tz=-3,
         mix=dict(hidro=0.52, gas=0.18, eolica=0.13, nuclear=0.09, solar=0.08)),
    dict(name="Madrid", country="España", region="Europa", role="Capital",
         pop_scope="área metropolitana", lat=40.42, lon=-3.70, pop_m=6.90, kwh_pc=5000, growth_pct=1.8,
         ev_share=0.09, dc_pipeline_mw=220, climate="mediterráneo continental", temp_mean=15.0, temp_amp_season=11.0,
         temp_amp_diurnal=10.0, heating_elec=0.15, cooling_sens=1.2, sectors=dict(res=0.36, com=0.40, ind=0.24),
         peak_factor=1.60, saidi_h=1.5, losses_pct=7.5, tz=1,
         mix=dict(eolica=0.24, nuclear=0.20, gas=0.19, solar=0.20, hidro=0.11, importaciones=0.06)),
    dict(name="Toronto", country="Canadá", region="América del Norte", role="Metrópoli (GTA)",
         pop_scope="área metropolitana", lat=43.65, lon=-79.38, pop_m=6.40, kwh_pc=8500, growth_pct=1.9,
         ev_share=0.08, dc_pipeline_mw=200, climate="continental húmedo", temp_mean=9.5, temp_amp_season=14.0,
         temp_amp_diurnal=7.0, heating_elec=0.45, cooling_sens=1.1, sectors=dict(res=0.38, com=0.38, ind=0.24),
         peak_factor=1.75, saidi_h=2.0, losses_pct=6.0, tz=-5,
         mix=dict(nuclear=0.53, hidro=0.22, eolica=0.09, gas=0.08, solar=0.05, importaciones=0.03)),
    dict(name="Singapur", country="Singapur", region="Asia", role="Ciudad-Estado",
         pop_scope="nacional urbana", lat=1.35, lon=103.82, pop_m=6.05, kwh_pc=8800, growth_pct=2.6,
         ev_share=0.10, dc_pipeline_mw=300, climate="ecuatorial", temp_mean=27.5, temp_amp_season=0.8,
         temp_amp_diurnal=5.0, heating_elec=0.0, cooling_sens=2.0, sectors=dict(res=0.28, com=0.42, ind=0.30),
         peak_factor=1.35, saidi_h=0.8, losses_pct=5.0, tz=8,
         mix=dict(gas=0.77, solar=0.09, importaciones=0.11, petroleo=0.03)),
    dict(name="Nairobi", country="Kenia", region="África", role="Capital",
         pop_scope="metropolitana", lat=-1.29, lon=36.82, pop_m=6.40, kwh_pc=550, growth_pct=5.0,
         ev_share=0.01, dc_pipeline_mw=40, climate="tropical de altura", temp_mean=19.0, temp_amp_season=2.0,
         temp_amp_diurnal=8.0, heating_elec=0.02, cooling_sens=0.6, sectors=dict(res=0.50, com=0.30, ind=0.20),
         peak_factor=1.45, saidi_h=18.0, losses_pct=15.0, tz=3,
         mix=dict(geotermica=0.44, hidro=0.29, eolica=0.13, solar=0.09, petroleo=0.05)),
    dict(name="Johannesburgo", country="Sudáfrica", region="África", role="Metrópoli económica",
         pop_scope="metropolitana", lat=-26.20, lon=28.05, pop_m=6.30, kwh_pc=5800, growth_pct=2.2,
         ev_share=0.02, dc_pipeline_mw=90, climate="subtropical de altura", temp_mean=16.5, temp_amp_season=8.0,
         temp_amp_diurnal=9.0, heating_elec=0.12, cooling_sens=0.9, sectors=dict(res=0.38, com=0.32, ind=0.30),
         peak_factor=1.60, saidi_h=10.0, losses_pct=11.5, tz=2,
         mix=dict(carbon=0.58, solar=0.11, eolica=0.09, nuclear=0.08, gas=0.08, hidro=0.06)),
    dict(name="Hanói", country="Vietnam", region="Asia", role="Capital",
         pop_scope="provincial urbana", lat=21.03, lon=105.85, pop_m=8.60, kwh_pc=2800, growth_pct=4.8,
         ev_share=0.03, dc_pipeline_mw=120, climate="subtropical húmedo", temp_mean=24.0, temp_amp_season=7.0,
         temp_amp_diurnal=5.5, heating_elec=0.05, cooling_sens=1.6, sectors=dict(res=0.44, com=0.30, ind=0.26),
         peak_factor=1.55, saidi_h=5.0, losses_pct=10.0, tz=7,
         mix=dict(hidro=0.30, carbon=0.33, gas=0.10, solar=0.12, eolica=0.08, importaciones=0.07)),
    dict(name="Casablanca", country="Marruecos", region="África", role="Intermedia / capital económica",
         pop_scope="metropolitana", lat=33.57, lon=-7.59, pop_m=7.10, kwh_pc=1100, growth_pct=3.2,
         ev_share=0.02, dc_pipeline_mw=50, climate="mediterráneo atlántico", temp_mean=18.5, temp_amp_season=6.5,
         temp_amp_diurnal=7.0, heating_elec=0.08, cooling_sens=1.0, sectors=dict(res=0.45, com=0.30, ind=0.25),
         peak_factor=1.50, saidi_h=8.0, losses_pct=11.0, tz=1,
         mix=dict(carbon=0.33, gas=0.19, eolica=0.21, solar=0.16, hidro=0.11)),
    dict(name="Dar es Salaam", country="Tanzania", region="África", role="Capital económica",
         pop_scope="metropolitana", lat=-6.79, lon=39.28, pop_m=7.40, kwh_pc=350, growth_pct=5.5,
         ev_share=0.005, dc_pipeline_mw=20, climate="tropical costero", temp_mean=27.0, temp_amp_season=2.5,
         temp_amp_diurnal=5.0, heating_elec=0.0, cooling_sens=1.5, sectors=dict(res=0.55, com=0.27, ind=0.18),
         peak_factor=1.45, saidi_h=25.0, losses_pct=16.0, tz=3,
         mix=dict(gas=0.40, hidro=0.34, solar=0.16, petroleo=0.10)),
]

CITY_NAMES = [c["name"] for c in CITIES]
_BY_NAME = {c["name"]: c for c in CITIES}

CLOUD_FACTOR = {  # fracción de cielo despejado efectiva por clima
    "ecuatorial": 0.45, "tropical monzónico": 0.48, "tropical sabana": 0.58,
    "tropical costero": 0.55, "tropical de altura": 0.60, "desértico costero": 0.72,
    "semiárido": 0.82, "mediterráneo": 0.78, "mediterráneo continental": 0.75,
    "mediterráneo atlántico": 0.72, "templado marítimo": 0.50,
    "continental húmedo": 0.60, "subtropical húmedo": 0.55,
    "subtropical de altura": 0.75,
}


def _rng(name: str, salt: str = "") -> np.random.Generator:
    h = hashlib.md5((name + salt).encode()).hexdigest()
    return np.random.default_rng(SEED_BASE + int(h[:8], 16))


def get_city(name: str) -> dict:
    return _BY_NAME[name]


def list_cities() -> list[str]:
    return CITY_NAMES


def renewable_share(mix: dict) -> float:
    return sum(v for k, v in mix.items() if k in ("hidro", "solar", "eolica", "geotermica", "biomasa"))


def grid_carbon_intensity(mix: dict) -> float:
    """Intensidad promedio (gCO2/kWh) de la matriz."""
    return sum(v * CO2_FACTORS.get(k, 450) for k, v in mix.items())


# ---------------------------------------------------------------------------
# Perfil horario base (forma diaria por sectores) — peso medio = 1
# ---------------------------------------------------------------------------
def _daily_shape(sectors: dict) -> np.ndarray:
    h = np.arange(24)
    res = 0.45 + 0.35 * np.exp(-((h - 7) ** 2) / 4.5) + 0.55 * np.exp(-((h - 19) ** 2) / 6.0)
    com = 0.30 + 0.95 * np.exp(-((h - 13.5) ** 2) / 12.0)
    ind = np.where((h >= 6) & (h < 22), 0.95, 0.70)
    return sectors["res"] * res + sectors["com"] * com + sectors["ind"] * ind


def _weekend_factor(sectors: dict) -> float:
    # ponderado: fin de semana baja comercio e industria
    return sectors["res"] * 1.05 + sectors["com"] * 0.55 + sectors["ind"] * 0.75


def _solar_pu(lat: float, doy: np.ndarray, hour_utc: np.ndarray, rng, cloud: float) -> np.ndarray:
    lat_r = np.radians(lat)
    decl = -np.radians(23.44) * np.cos(2 * np.pi * (doy + 10) / 365.25)
    ha = np.radians(15.0 * (hour_utc - 12.0))
    sin_elev = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    clear = np.clip(sin_elev, 0, None) ** 0.85
    noise = 0.80 + 0.25 * rng.random(clear.shape)
    return np.clip(clear * cloud * noise, 0, 1)


def _wind_pu(doy: np.ndarray, hour_utc: np.ndarray, rng, base: float = 0.28) -> np.ndarray:
    seasonal = 0.10 * np.sin(2 * np.pi * (doy - 40) / 365.25)
    diurnal = 0.05 * np.sin(2 * np.pi * (hour_utc - 3) / 24.0)
    noise = 0.10 * (rng.standard_normal(doy.shape))
    return np.clip(base + seasonal + diurnal + noise, 0.0, 1.0)


def _temperature(city: dict, doy: np.ndarray, hour_local: np.ndarray, rng) -> np.ndarray:
    hemi = -1.0 if city["lat"] < 0 else 1.0
    season = city["temp_amp_season"] * np.sin(2 * np.pi * (doy - 105) / 365.25) * hemi
    diurnal = city["temp_amp_diurnal"] * 0.5 * np.sin(2 * np.pi * (hour_local - 9) / 24.0)
    n = rng.standard_normal(len(doy))
    ar = np.zeros_like(n)
    ar[0] = n[0]
    for i in range(1, len(n)):  # AR(1) ligero
        ar[i] = 0.85 * ar[i - 1] + 0.52 * n[i]
    return city["temp_mean"] + season + diurnal + ar


def city_hourly(name: str, years: int = 2, end_year: int = 2026) -> pd.DataFrame:
    """Serie horaria sintética completa de una ciudad (determinística)."""
    c = get_city(name)
    rng = _rng(name, "hourly")
    idx = pd.date_range(f"{end_year - years}-01-01", f"{end_year}-01-01", freq="h", inclusive="left")
    n = len(idx)
    doy = idx.dayofyear.to_numpy().astype(float)
    hour_utc = idx.hour.to_numpy().astype(float)
    hour_local = (hour_utc + c["tz"]) % 24
    is_weekend = (idx.dayofweek.to_numpy() >= 5).astype(float)
    t_years = np.arange(n) / 8760.0

    temp = _temperature(c, doy, hour_local, rng)
    cloud = CLOUD_FACTOR.get(c["climate"], 0.6)
    solar = _solar_pu(c["lat"], doy, hour_utc, rng, cloud)
    wind = _wind_pu(doy, hour_utc, rng)

    base_avg_mw = c["pop_m"] * 1e6 * c["kwh_pc"] / 8760.0 / 1000.0  # MW promedio
    shape = _daily_shape(c["sectors"])
    day_w = shape[np.clip(hour_local.astype(int), 0, 23)]
    day_w = day_w / day_w.mean()
    wk_factor = 1.0 + is_weekend * (_weekend_factor(c["sectors"]) - 1.0)

    # efecto temperatura (refrigeración / calefacción eléctrica)
    cool = np.clip(temp - 22.0, 0, None) * 0.022 * c["cooling_sens"]
    heat = np.clip(15.0 - temp, 0, None) * 0.020 * c["heating_elec"] * 3.0
    temp_eff = 1.0 + cool + heat

    trend = (1.0 + c["growth_pct"] / 100.0) ** t_years

    # ruido AR(1)
    nn = rng.standard_normal(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.90 * ar[i - 1] + 0.44 * nn[i]
    noise = 1.0 + 0.025 * ar

    # carga de vehículos eléctricos (tarde-noche, crece con la serie)
    ev_shape = np.exp(-((hour_local - 20) ** 2) / 6.0) * (1 - is_weekend * 0.25)
    ev_mw = (base_avg_mw * 0.06 * c["ev_share"] * 10.0) * ev_shape * (1 + 0.30 * t_years)

    # carga de data centers (casi plana, crece 18 %/año desde pipeline inicial 60 %)
    dc_shape = 0.85 + 0.15 * np.sin(2 * np.pi * (hour_local - 4) / 24.0)
    dc_mw = c["dc_pipeline_mw"] * (0.60 + 0.40 * t_years / max(years, 1)) * (1.18 ** t_years) * dc_shape

    demand = base_avg_mw * trend * day_w * wk_factor * temp_eff * noise + ev_mw + dc_mw

    # intensidad de carbono instantánea: la solar/eólica reduce el residual fósil
    ci_base = grid_carbon_intensity(c["mix"])
    ren_inst = (c["mix"].get("solar", 0) * solar * 2.2 + c["mix"].get("eolica", 0) * wind * 1.6)
    ci = np.clip(ci_base * (1.0 - 0.55 * np.clip(ren_inst, 0, 0.6)) + rng.normal(0, 8, n), 20, None)

    # precio sintético (merit-order suave sobre demanda normalizada)
    d_norm = demand / np.mean(demand)
    dn = d_norm - 1.0
    signed_pow = np.sign(dn) * np.abs(dn) ** 1.6
    price = np.clip(38 + 55 * signed_pow * 2.2 + 0.15 * ci_base
                    + rng.normal(0, 6, n) + 12 * np.clip(temp - 26, 0, None), 8, 400)

    df = pd.DataFrame({
        "demand_mw": demand,
        "ev_mw": ev_mw,
        "dc_mw": dc_mw,
        "temp_c": temp,
        "solar_pu": solar,
        "wind_pu": wind,
        "ci_gco2_kwh": ci,
        "price_eur_mwh": price,
    }, index=idx)
    df.index.name = "time"
    df.attrs["city"] = c
    return df


def city_monthly_10y(name: str, end_year: int = 2026) -> pd.DataFrame:
    """Agregado mensual de 10 años para tendencias (rápido de generar)."""
    c = get_city(name)
    rng = _rng(name, "monthly")
    idx = pd.period_range(f"{end_year-10}-01", f"{end_year}-01", freq="M").to_timestamp()
    n = len(idx)
    base_avg_mw = c["pop_m"] * 1e6 * c["kwh_pc"] / 8760.0 / 1000.0
    t_years = np.arange(n) / 12.0
    doy_m = idx.month.to_numpy()
    season = 1.0 + 0.06 * np.sin(2 * np.pi * (doy_m - 3) / 12.0) * (1 if c["lat"] >= 0 else -1)
    trend = (1 + c["growth_pct"] / 100) ** t_years
    dc = c["dc_pipeline_mw"] * (0.35 + 0.65 * t_years / 10.0) * (1.18 ** t_years)
    ev = base_avg_mw * 0.06 * c["ev_share"] * 10.0 * (0.4 + 0.6 * t_years / 10.0)
    dem = base_avg_mw * trend * season + dc + ev + rng.normal(0, base_avg_mw * 0.01, n)
    return pd.DataFrame({
        "demand_mw_avg": dem,
        "dc_mw": dc,
        "ev_mw": ev,
        "pop_m": c["pop_m"] * (1 + 0.009) ** t_years,  # crecimiento demográfico urbano ~0.9 %/año
    }, index=idx)


def city_summary_table() -> pd.DataFrame:
    """Tabla resumen del catálogo (para mapa y tabla comparativa)."""
    rows = []
    for c in CITIES:
        df = None
        base_avg_mw = c["pop_m"] * 1e6 * c["kwh_pc"] / 8760.0 / 1000.0
        peak_mw = base_avg_mw * c["peak_factor"] + c["dc_pipeline_mw"] * 0.6
        rows.append(dict(
            ciudad=c["name"], pais=c["country"], region=c["region"], rol=c["role"],
            alcance_pob=c["pop_scope"], poblacion_M=c["pop_m"], lat=c["lat"], lon=c["lon"],
            demanda_TWh_año=round(c["pop_m"] * c["kwh_pc"] / 1000.0, 2),
            pico_MW=round(peak_mw, 0), kwh_per_capita=c["kwh_pc"],
            crecimiento_pct=c["growth_pct"], share_VE_pct=round(c["ev_share"] * 100, 1),
            datacenter_MW=c["dc_pipeline_mw"],
            renovables_pct=round(renewable_share(c["mix"]) * 100, 1),
            ci_gCO2_kwh=round(grid_carbon_intensity(c["mix"]), 0),
            saidi_h=c["saidi_h"], perdidas_pct=c["losses_pct"], clima=c["climate"],
        ))
    return pd.DataFrame(rows)


def city_feature_matrix() -> pd.DataFrame:
    """Matriz de indicadores urbanos para PCA (OE-1, aprendizaje no supervisado)."""
    t = city_summary_table().set_index("ciudad")
    feats = t[["poblacion_M", "demanda_TWh_año", "kwh_per_capita", "crecimiento_pct",
               "share_VE_pct", "datacenter_MW", "renovables_pct", "ci_gCO2_kwh",
               "saidi_h", "perdidas_pct", "pico_MW"]].copy()
    feats["region"] = t["region"]
    return feats


def daily_profiles(name: str, years: int = 2) -> pd.DataFrame:
    """Perfiles diarios normalizados (24 valores) para clustering K-Means."""
    df = city_hourly(name, years=years)
    d = df["demand_mw"].groupby(df.index.date).mean()
    hourly = df["demand_mw"].groupby([df.index.date, df.index.hour]).mean().unstack()
    hourly = hourly.div(hourly.mean(axis=1), axis=0)
    hourly.index = pd.to_datetime(hourly.index)
    return hourly

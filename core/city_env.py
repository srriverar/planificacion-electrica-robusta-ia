"""Grid2Op *por ciudad*: construye un ambiente Grid2Op dedicado para cada una de las
16 ciudades, donde las **crónicas** (series de inyección) se re-forman a partir de la
curva de demanda sintética de la ciudad (factor pico, mezcla renovable, crecimiento).

La red física es la misma malla didáctica `rte_case5_example` (5 subestaciones,
3 cargas, 2 generadores, 8 líneas): un banco de pruebas estándar de L2RPN. Lo que
cambia **por ciudad** es la crónica temporal que alimenta la red — su forma diaria,
su amplitud de pico y su penetración renovable — de modo que el agente DQN entrenado
se enfrenta a la "personalidad" operativa de cada ciudad. Los valores absolutos se
mantienen en el rango de la malla (≈MW) para que los pesos entrenados transfieran.

Todo es determinístico (semillas por ciudad) y se cachea en disco.
"""
from __future__ import annotations

import bz2
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import core.synthetic_data as sd

BASE_ENV = os.path.join(
    os.path.dirname(os.path.dirname(__import__("grid2op").__file__)),
    "grid2op", "data", "rte_case5_example",
)
# Fallback explícito a la ruta del paquete instalado
if not os.path.isdir(BASE_ENV):  # pragma: no cover
    import grid2op as _g2
    BASE_ENV = os.path.join(os.path.dirname(os.path.abspath(_g2.__file__)),
                            "data", "rte_case5_example")

CACHE_DIR = os.path.join(tempfile.gettempdir(), "g2o_city_envs_v1")
STEPS = 2016          # una semana a 5 min (7*288)
N_CHRONICS_CITY = 5   # variantes (semanas) por ciudad

# Estadísticas de referencia de la malla case5 (MW) usadas para preservar magnitudes
BASE_LOAD_MEAN = np.array([9.07, 8.22, 8.42])   # 3 cargas
GEN1_MEAN, GEN1_MAX = 22.68, 30.4               # generador despachable
GEN0_MAX = 9.2                                  # generador renovable (techo ~10 MW)


def _seed(name: str, salt: str = "") -> int:
    return int(hashlib.md5((name + "|" + salt).encode()).hexdigest()[:8], 16)


def _city_weeks(name: str, n: int) -> list[np.ndarray]:
    """Devuelve `n` perfiles semanales (168 h) de demanda normalizada (media≈1),
    centrados en distintos picos representativos de la ciudad."""
    df = sd.city_hourly(name, years=1)
    dem = df["demand_mw"].to_numpy(dtype=float)
    mean = dem.mean()
    pu = dem / mean
    H = len(pu)
    out = []
    # semana del pico anual + semanas distribuidas por la temporada
    peak_h = int(np.argmax(pu))
    offsets = [max(0, peak_h - 84)]
    for k in range(1, n):
        offsets.append(int((k / n) * (H - 168)))
    for off in offsets[:n]:
        wk = pu[off:off + 168]
        if len(wk) < 168:  # envolver si se pasa del final del año
            wk = np.concatenate([wk, pu[:168 - len(wk)]])
        out.append(wk / max(wk.mean(), 1e-6))
    return out


def _city_renewable_weeks(name: str, n: int) -> list[np.ndarray]:
    """Perfil semanal renovable normalizado (solar+viento) por ciudad, 0..1."""
    df = sd.city_hourly(name, years=1)
    solar = df["solar_pu"].to_numpy(dtype=float)
    wind = df["wind_pu"].to_numpy(dtype=float)
    mix = sd.get_city(name)["mix"]
    w_sol = mix.get("solar", 0.0)
    w_win = mix.get("eolica", mix.get("wind", 0.0))
    denom = max(w_sol + w_win, 1e-6)
    renew = (solar * w_sol + wind * w_win) / denom
    H = len(renew)
    peak_h = int(np.argmax(df["demand_mw"].to_numpy()))
    offsets = [max(0, peak_h - 84)] + [int((k / n) * (H - 168)) for k in range(1, n)]
    out = []
    for off in offsets[:n]:
        wk = renew[off:off + 168]
        if len(wk) < 168:
            wk = np.concatenate([wk, renew[:168 - len(wk)]])
        m = wk.max()
        out.append(wk / m if m > 1e-6 else np.full(168, 0.5))
    return out


def _upsample(weekly_hourly: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """De 168 h a `steps` pasos de 5 min (interp. lineal). 168→2016 = factor 12."""
    n_h = len(weekly_hourly)
    xs = np.linspace(0.0, 1.0, n_h, endpoint=False)
    xt = np.linspace(0.0, 1.0, steps, endpoint=False)
    # repetir el ciclo para continuidad
    xs2 = np.concatenate([xs, [1.0]])
    y2 = np.concatenate([weekly_hourly, [weekly_hourly[0]]])
    return np.interp(xt, xs2, y2)


def _build_chronic_arrays(name: str, variant: int) -> dict[str, np.ndarray]:
    """Construye load_p/load_q/prod_p/prod_v (2016 pasos) para una variante de ciudad."""
    c = sd.get_city(name)
    ren_share = sd.renewable_share(c["mix"])           # 0..1
    peak_factor = float(c.get("peak_factor", 1.3))     # >1 => más picuda
    growth = float(c.get("growth_pct", 1.0))           # % anual

    dem_wk = _city_weeks(name, N_CHRONICS_CITY)[variant]
    ren_wk = _city_renewable_weeks(name, N_CHRONICS_CITY)[variant]

    dem_pu = _upsample(dem_wk, STEPS)                  # media≈1, picos según ciudad
    ren_pu = _upsample(ren_wk, STEPS)                  # 0..1

    # acentuar la forma según el factor pico (ciudad picuda => mayor swing relativo)
    gain = 1.0 + 0.35 * (peak_factor - 1.0)
    dem_pu = 1.0 + (dem_pu - 1.0) * gain
    # tendencia de crecimiento ligera a lo largo de la semana (proxy intra-semanal)
    trend = np.linspace(1.0, 1.0 + growth / 100.0 * 0.02, STEPS)
    dem_pu = dem_pu * trend
    # techo absoluto del pico total (MW): la malla case5 solo cubre ~30 MW despachables
    # (gen1_max≈30.4). Ciudades más picudas => pico total más alto => más difícil,
    # pero siempre dentro de un rango sobrevivible para que el DQN entrenado transfiera.
    peak_ratio = max(float(dem_pu.max()), 1e-6)
    target_peak_total = 26.0 + 3.5 * min(1.0, max(0.0, (peak_factor - 1.0)) / 0.6)
    scale = target_peak_total / (BASE_LOAD_MEAN.sum() * peak_ratio)
    dem_pu = dem_pu * scale

    rng = np.random.default_rng(_seed(name, f"chr{variant}"))
    # 3 cargas: comparten la forma con ligeras diferencias + ruido suave
    noise = [rng.normal(1.0, 0.02, STEPS) for _ in range(3)]
    load_p = np.clip(BASE_LOAD_MEAN[:, None] * dem_pu[None, :] * np.array(noise), 2.0, None)
    load_q = load_p * 0.70

    # renovable: amplitud según participación renovable de la ciudad (techo GEN0_MAX)
    amp = GEN0_MAX * min(1.0, ren_share / 0.55)
    gen0 = np.clip(amp * ren_pu, 0.0, 10.0)
    # despachable cubre el resto + margen (~2% de pérdidas), limitado a 30 MW
    total_load = load_p.sum(axis=0)
    gen1 = np.clip(total_load * 1.02 - gen0, 0.0, GEN1_MAX)

    prod_p = np.vstack([gen0, gen1]).T                 # columnas gen_0_0, gen_1_1
    prod_v = np.full((STEPS, 2), 102.0)

    return dict(
        load_p=load_p.T, load_q=load_q.T, prod_p=prod_p, prod_v=prod_v,
    )


def _write_csv_bz2(path: str, df: pd.DataFrame) -> None:
    tmp = path[:-4] if path.endswith(".bz2") else path
    df.to_csv(tmp, sep=";", index=False, float_format="%.4f")
    if path.endswith(".bz2"):
        with open(tmp, "rb") as fh, bz2.BZ2File(path, "wb") as out:
            shutil.copyfileobj(fh, out)
        os.remove(tmp)


def build_city_env(name: str, force: bool = False) -> str:
    """Crea (y cachea) la carpeta de ambiente Grid2Op de una ciudad. Devuelve la ruta."""
    slug = hashlib.md5(name.encode()).hexdigest()[:10]
    dst = os.path.join(CACHE_DIR, f"city_{slug}")
    marker = os.path.join(dst, ".built")
    if os.path.exists(marker) and not force:
        return dst

    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)

    # copiar la red física y config (todo menos chronics)
    for item in os.listdir(BASE_ENV):
        if item in ("chronics", "__pycache__"):
            continue
        s = os.path.join(BASE_ENV, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # chronics: N_CHRONICS_CITY variantes con la forma de la ciudad
    chron_root = os.path.join(dst, "chronics")
    os.makedirs(chron_root, exist_ok=True)
    line_cols = "0_1_0;0_2_1;0_3_2;0_4_3;1_2_4;2_3_5;2_3_6;3_4_7".split(";")
    load_cols = ["load_0_0", "load_3_1", "load_4_2"]
    gen_cols = ["gen_0_0", "gen_1_1"]
    zeros_lines = np.zeros((STEPS, len(line_cols)))

    for v in range(N_CHRONICS_CITY):
        cdir = os.path.join(chron_root, f"{v:02d}")
        os.makedirs(cdir, exist_ok=True)
        arr = _build_chronic_arrays(name, v)
        _write_csv_bz2(os.path.join(cdir, "load_p.csv.bz2"), pd.DataFrame(arr["load_p"], columns=load_cols))
        _write_csv_bz2(os.path.join(cdir, "load_q.csv.bz2"), pd.DataFrame(arr["load_q"], columns=load_cols))
        _write_csv_bz2(os.path.join(cdir, "prod_p.csv.bz2"), pd.DataFrame(arr["prod_p"], columns=gen_cols))
        _write_csv_bz2(os.path.join(cdir, "prod_v.csv.bz2"), pd.DataFrame(arr["prod_v"], columns=gen_cols))
        # forecasts = copia (el DQN no usa simulate, pero evitamos warnings)
        for base in ("load_p", "load_q", "prod_p", "prod_v"):
            shutil.copy2(os.path.join(cdir, f"{base}.csv.bz2"),
                         os.path.join(cdir, f"{base}_forecasted.csv.bz2"))
        _write_csv_bz2(os.path.join(cdir, "maintenance.csv.bz2"), pd.DataFrame(zeros_lines, columns=line_cols))
        _write_csv_bz2(os.path.join(cdir, "hazards.csv.bz2"), pd.DataFrame(zeros_lines, columns=line_cols))
        _write_csv_bz2(os.path.join(cdir, "maintenance_forecasted.csv.bz2"), pd.DataFrame(zeros_lines, columns=line_cols))
        with open(os.path.join(cdir, "time_interval.info"), "w") as fh:
            fh.write("00:05\n")
        with open(os.path.join(cdir, "start_datetime.info"), "w") as fh:
            fh.write((datetime(2026, 1, 5) + timedelta(days=7 * v)).strftime("%Y-%m-%d %H:%M") + "\n")

    with open(marker, "w") as fh:
        fh.write("ok\n")
    return dst


def make_city_env(name: str, force: bool = False):
    """Devuelve un ambiente Grid2Op listo para la ciudad dada."""
    import warnings
    import grid2op
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = build_city_env(name, force=force)
        env = grid2op.make(path)
    return env


def city_env_summary(name: str) -> dict:
    """Resumen de la crónica de la ciudad (para mostrar en la UI, sin correr el env)."""
    c = sd.get_city(name)
    dem_wk = _city_weeks(name, N_CHRONICS_CITY)[0]
    dem_pu = _upsample(dem_wk, STEPS)
    pf = float(c.get("peak_factor", 1.3))
    gain = 1.0 + 0.35 * (pf - 1.0)
    dem_pu = 1.0 + (dem_pu - 1.0) * gain
    dem_pu *= np.linspace(1.0, 1.0 + float(c.get("growth_pct", 1.0)) / 100.0 * 0.02, STEPS)
    peak_ratio = max(float(dem_pu.max()), 1e-6)
    target_peak_total = 26.0 + 3.5 * min(1.0, max(0.0, (pf - 1.0)) / 0.6)
    scale = target_peak_total / (BASE_LOAD_MEAN.sum() * peak_ratio)
    total = BASE_LOAD_MEAN.sum() * dem_pu * scale
    return dict(
        ren_share=round(sd.renewable_share(c["mix"]), 3),
        peak_factor=round(float(c.get("peak_factor", 1.3)), 3),
        growth_pct=round(float(c.get("growth_pct", 1.0)), 2),
        peak_load_mw=round(float(total.max()), 2),
        mean_load_mw=round(float(total.mean()), 2),
        min_load_mw=round(float(total.min()), 2),
        n_steps=STEPS, n_chronics=N_CHRONICS_CITY,
    )

"""Descarga los **generadores solares de OpenStreetMap** (power=generator con
generator:source solar/photovoltaic) dentro del límite de cada ciudad.

Salida: data/osm_solar/{slug}.csv con columnas
    latitude, longitude, mwp, name, osm_id, osm_type

Es la fuente de respaldo mundial para la capa de glow solar cuando el país no
tiene atlas EarthPV (junto con el subconjunto solar de WRI GPPD).

Uso:
    python scripts/fetch_osm_solar.py            # 16 ciudades
    python scripts/fetch_osm_solar.py Bogotá     # una ciudad
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

GRID2POSTER_DIR = os.environ.get("GRID2POSTER_DIR", "/home/user/grid2poster")
sys.path.insert(0, GRID2POSTER_DIR)
OUT_DIR = os.path.join(ROOT, "data", "osm_solar")

# Supuesto divulgado (S10.7): generador solar sin etiqueta de potencia → 20 kWp
# (instalación residencial/comercial típica) para que aporte al glow.
DEFAULT_ROOFTOP_MWP = 0.02


def fetch_city_solar(city: str) -> dict:
    from render_city_maps import _geocode_query, _install_overpass_retry
    _install_overpass_retry()
    import requests
    from shapely.ops import unary_union
    from osm_data import (get_country_boundary, _polygon_to_overpass_poly,
                          _simplify_boundary_for_overpass)
    from prepare import parse_capacity_to_mw
    from common import slugify

    query = _geocode_query(city)
    boundary = get_country_boundary(query, use_cache=True)
    geom = unary_union(boundary.geometry)
    polys = _simplify_boundary_for_overpass(geom)
    poly_filters = "".join(
        f'nwr["power"="generator"]["generator:source"~"solar|photovoltaic",i]'
        f'(poly:"{_polygon_to_overpass_poly(p)}");'
        for p in polys
    )
    ql = f'[out:json][timeout:600];({poly_filters});out center tags;'

    r = None
    mirrors = ["https://overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://overpass.private.coffee/api/interpreter",
               "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
    for attempt in range(16):  # reintenta también 406/400 transitorios
        mirror = mirrors[attempt % len(mirrors)]
        try:
            resp = requests.post(mirror, data={"data": ql}, timeout=660)
            if resp.status_code == 200:
                r = resp
                break
            print(f"    intento {attempt+1}: HTTP {resp.status_code} @ {mirror}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    intento {attempt+1}: {type(e).__name__} @ {mirror}", flush=True)
        time.sleep(4 + attempt * 3)
    if r is None:
        raise RuntimeError("Overpass agotó los reintentos (solar)")
    elements = r.json().get("elements", [])

    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        src = str(tags.get("generator:source", "")).lower()
        if "solar" not in src and "photovoltaic" not in src:
            continue
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        mw = parse_capacity_to_mw(tags.get("generator:output:electricity"))
        rows.append({
            "latitude": float(lat), "longitude": float(lon),
            "mwp": float(mw) if mw and mw > 0 else DEFAULT_ROOFTOP_MWP,
            "mwp_tagged": 1 if (mw and mw > 0) else 0,
            "name": tags.get("name", ""),
            "osm_id": el.get("id"), "osm_type": el.get("type"),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"{slugify(city)}.csv")
    import pandas as pd
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "mwp", "mwp_tagged",
                                     "name", "osm_id", "osm_type"])
    df.to_csv(out_csv, index=False)
    return {"city": city, "n": len(df), "mwp": round(float(df.mwp.sum()), 2) if len(df) else 0.0,
            "tagged": int(df.mwp_tagged.sum()) if len(df) else 0, "csv": out_csv}


def main(cities: list[str]) -> None:
    for i, city in enumerate(cities, 1):
        t0 = time.time()
        try:
            st = fetch_city_solar(city)
            print(f"[{i:2d}/{len(cities)}] {city:15s} -> OK {st['n']:5d} generadores solares, "
                  f"{st['mwp']:9.1f} MWp ({st['tagged']} con etiqueta) en {time.time()-t0:.0f}s",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:2d}/{len(cities)}] {city:15s} -> EXCEPCIÓN {type(e).__name__}: {str(e)[:140]}",
                  flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    import core.synthetic_data as sd
    main(args if args else sd.list_cities())

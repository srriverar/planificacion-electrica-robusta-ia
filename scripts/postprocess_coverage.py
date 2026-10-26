"""Calcula, para cada ciudad, los **km reales de líneas de transmisión mapeadas en OSM**
por nivel de tensión, reusando la caché de grid2poster (sin re-descargar). Escribe
assets/city_maps/coverage.json, que la pestaña 🗺️ muestra como KPI de cobertura.

Correr DESPUÉS de scripts/render_city_maps.py (para que la caché ya tenga los datos).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
G2P = os.environ.get("GRID2POSTER_DIR", "/home/user/grid2poster")
sys.path.insert(0, G2P)

CRS = "EPSG:3857"
BUCKETS = [("<60 kV", 0, 60), ("60–150 kV", 60, 150),
           ("150–300 kV", 150, 300), (">300 kV", 300, 1e9)]  # sort_voltage viene en kV


def _utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def main():
    import core.synthetic_data as sd
    from osm_data import get_country_boundary, fetch_power_features_single
    from prepare import prepare_lines

    out = {}
    cities = sd.list_cities()
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "rcm", os.path.join(ROOT, "scripts", "render_city_maps.py"))
    _rcm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_rcm)          # solo definiciones; main() está protegido
    _geocode_query = _rcm._geocode_query    # respeta GEOCODE_OVERRIDES
    for i, city in enumerate(cities, 1):
        c = sd.get_city(city)
        query = _geocode_query(city)
        try:
            b = get_country_boundary(query, use_cache=True)
            raw = fetch_power_features_single(query, b, use_cache=True)
            lines = prepare_lines(raw, b, CRS)
            if lines is None or len(lines) == 0:
                out[city] = dict(total_km=0.0, n_lines=0, tiers={})
                print(f"[{i:2d}] {city:15s} sin líneas", flush=True)
                continue
            g = lines.to_crs(_utm_epsg(c["lat"], c["lon"]))
            g = g.assign(km=g.geometry.length / 1000.0)
            v = g["sort_voltage"].fillna(0).to_numpy()
            tiers = {}
            for name, lo, hi in BUCKETS:
                km = float(g.loc[(v >= lo) & (v < hi), "km"].sum())
                if km > 0.5:
                    tiers[name] = round(km, 1)
            out[city] = dict(total_km=round(float(g["km"].sum()), 1),
                             n_lines=int(len(g)), tiers=tiers)
            print(f"[{i:2d}] {city:15s} {out[city]['total_km']:8.1f} km en {out[city]['n_lines']} tramos {tiers}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:2d}] {city:15s} ERROR {type(e).__name__}: {str(e)[:100]}", flush=True)

    path = os.path.join(ROOT, "assets", "city_maps", "coverage.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nGuardado: {path}", flush=True)


if __name__ == "__main__":
    main()

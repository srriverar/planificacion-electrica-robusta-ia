"""Estadísticas de las capas enriquecidas para la pestaña 🗺️.

Genera (reusando las cachés de grid2poster en cache/):
  assets/city_maps/plants_stats.json  → centrales OSM (power=plant) por ciudad
  assets/city_maps/solar_stats.json   → capa solar (EarthPV / OSM+GPPD) por ciudad

Ejecutar DESPUÉS de scripts/render_city_maps_layers.py (para que la caché de
power=plant exista) y de scripts/fetch_osm_solar.py.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
GRID2POSTER_DIR = os.environ.get("GRID2POSTER_DIR", "/home/user/grid2poster")
sys.path.insert(0, GRID2POSTER_DIR)
PV_DIR = os.environ.get("PV_DIR", "/home/user/pv_layers")
OUT = os.path.join(ROOT, "assets", "city_maps")

from render_city_maps import _geocode_query, _install_overpass_retry  # noqa: E402
from render_city_maps_layers import EARTHPV_BY_CITY, build_merged_solar_csv  # noqa: E402

TILE_KM, CRS = 400.0, "EPSG:3857"   # defaults de grid2poster (coinciden con su caché)
PV_TOLERANCE_M = 5000.0             # mismo margen que pv_data.PV_BOUNDARY_TOLERANCE_M


def _city_mask(boundary, crs=CRS):
    from shapely.ops import unary_union
    import shapely
    bp = boundary.to_crs(crs)
    mask = unary_union(bp.geometry).buffer(PV_TOLERANCE_M)
    shapely.prepare(mask)
    return mask


def _clip_points(layer, mask):
    """layer: GeoDataFrame WGS84 con columna 'mwp'. Devuelve máscara de puntos dentro."""
    import shapely
    pts = layer.to_crs(CRS).geometry.values
    return shapely.contains(mask, pts)


def plants_for_city(city, boundary):
    from osm_data import fetch_power_plants
    from prepare import prepare_plants
    raw = fetch_power_plants(country=_geocode_query(city), boundary=boundary,
                             tile_size_km=TILE_KM, render_crs=CRS, use_cache=True)
    pp = prepare_plants(raw, boundary, CRS)
    if pp is None or pp.empty:
        return {"n": 0, "total_mw": 0.0, "known_mw": 0.0, "n_known": 0, "by_source": {}, "top": []}
    pp = pp.copy()
    pp["cap"] = pp["capacity_mw"].fillna(0.0)
    by_source = {}
    for b, g in pp.groupby("source_bucket"):
        by_source[str(b)] = {"n": int(len(g)), "mw": round(float(g["cap"].sum()), 1)}
    pp["_name"] = (pp["name"].fillna("").astype(str) if "name" in pp.columns else "")
    h = pp.sort_values("cap", ascending=False).head(10)
    top = [{"name": (r["_name"] or "(sin nombre)"),
            "source": str(r.get("source_bucket", "other")),
            "mw": round(float(r["cap"]), 1)} for _, r in h.iterrows()]
    return {"n": int(len(pp)),
            "total_mw": round(float(pp["cap"].sum()), 1),
            "known_mw": round(float(pp.loc[pp["cap"] > 0, "cap"].sum()), 1),
            "n_known": int((pp["cap"] > 0).sum()),
            "by_source": by_source, "top": top}


def solar_for_city(city, boundary, mask, slug, query):
    if city in EARTHPV_BY_CITY:
        from render_city_maps_layers import earthpv_city_layer
        clipped, pais, tier = earthpv_city_layer(city, slug, query)
        pq = EARTHPV_BY_CITY[city][0]
        from pv_data import load_pv_layer
        import pandas as pd
        layer = load_pv_layer(clipped, use_cache=False)
        nat = float(pd.read_parquet(os.path.join(PV_DIR, pq))["mwp_best"].sum())
        inside = _clip_points(layer, mask)
        return {"kind": "earthpv", "pais": pais, "tier": tier, "archivo": pq,
                "n_features": int(len(layer)),
                "mwp_nacional": round(nat, 1),
                "n_ciudad": int(inside.sum()),
                "mwp_ciudad": round(float(layer.loc[inside, "mwp"].sum()), 2),
                "desglose": {"earthpv": round(float(layer.loc[inside, "mwp"].sum()), 2)}}
    merged = build_merged_solar_csv(slug)
    if not merged:
        return {"kind": "sin_datos", "n_features": 0, "mwp_nacional": 0.0,
                "n_ciudad": 0, "mwp_ciudad": 0.0, "desglose": {}}
    import pandas as pd
    import geopandas as gpd
    df = pd.read_csv(merged)
    layer = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude),
                             crs="EPSG:4326")
    layer = layer[layer["mwp"] > 0]
    inside = _clip_points(layer, mask)
    d = layer[inside]
    desglose = {s: round(float(g["mwp"].sum()), 2) for s, g in d.groupby("source")}
    n_por = {s: int(len(g)) for s, g in d.groupby("source")}
    return {"kind": "osm+gppd", "pais": None, "tier": None, "archivo": "osm_solar + gppd_solar",
            "n_features": int(len(layer)),
            "mwp_nacional": None,
            "n_ciudad": int(len(d)),
            "mwp_ciudad": round(float(d["mwp"].sum()), 2),
            "desglose": desglose, "n_desglose": n_por}


def main(cities):
    _install_overpass_retry()
    import core.synthetic_data as sd
    from common import slugify
    from osm_data import get_country_boundary

    plants, solar = {}, {}
    for i, city in enumerate(cities, 1):
        try:
            boundary = get_country_boundary(_geocode_query(city), use_cache=True)
            mask = _city_mask(boundary)
            plants[city] = plants_for_city(city, boundary)
            solar[city] = solar_for_city(city, boundary, mask, slugify(city),
                                         _geocode_query(city))
            p, s = plants[city], solar[city]
            print(f"[{i:2d}/{len(cities)}] {city:15s} plantas {p['n']:4d} ({p['total_mw']:8.1f} MW) | "
                  f"solar {s['kind']:9s} {s['n_ciudad']:4d} feat {s['mwp_ciudad']:9.2f} MWp", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:2d}/{len(cities)}] {city:15s} -> EXCEPCIÓN {type(e).__name__}: {str(e)[:120]}",
                  flush=True)

    with open(os.path.join(OUT, "plants_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(plants, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "solar_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(solar, fh, ensure_ascii=False, indent=1)
    print("Guardados: plants_stats.json, solar_stats.json en", OUT, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    import core.synthetic_data as sd
    main(args if args else sd.list_cities())

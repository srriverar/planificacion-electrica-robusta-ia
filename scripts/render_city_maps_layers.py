"""Renderiza las variantes ENRIQUECIDAS del mapa de cada ciudad con grid2poster:

  plantas/  → --show-plants: centrales eléctricas de OSM (power=plant) como
              marcadores dimensionados por capacidad y coloreados por fuente.
  solar/    → glow solar bajo la red: atlas EarthPV (Casablanca, Johannesburgo,
              Hanói) o capa combinada OSM(generadores solares)+WRI GPPD (resto).

Salidas: assets/city_maps/plantas/{slug}.png y assets/city_maps/solar/{slug}.png.

Uso:
    python scripts/render_city_maps_layers.py                 # ambas, 16 ciudades
    python scripts/render_city_maps_layers.py solar Bogotá    # variante y ciudades
Requiere: scripts/fetch_osm_solar.py ejecutado antes (para data/osm_solar/*.csv)
y las capas en /home/user/pv_layers/ (earthpv_*.parquet, gppd_solar.csv).
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
PV_DIR = os.environ.get("PV_DIR", "/home/user/pv_layers")
OUT_ROOT = os.path.join(ROOT, "assets", "city_maps")
OSM_SOLAR_DIR = os.path.join(ROOT, "data", "osm_solar")
MERGED_DIR = os.path.join(ROOT, "data", "solar_merged")
THEME = os.environ.get("G2P_THEME", "paper_grid")
WIDTH_MM, HEIGHT_MM, DPI = 200, 260, 100

# Ciudades cuyo país tiene atlas EarthPV publicado (tier entre paréntesis)
EARTHPV_BY_CITY = {
    "Casablanca": ("earthpv_morocco.parquet", "Marruecos", "Bronze"),
    "Johannesburgo": ("earthpv_south_africa.parquet", "Sudáfrica", "Bronze"),
    "Hanói": ("earthpv_vietnam.parquet", "Vietnam", "Bronze"),
}


def build_merged_solar_csv(slug: str) -> str | None:
    """Une generadores solares OSM (si existen) + subconjunto solar de WRI GPPD."""
    import pandas as pd
    frames = []
    osm_csv = os.path.join(OSM_SOLAR_DIR, f"{slug}.csv")
    if os.path.exists(osm_csv):
        try:
            d = pd.read_csv(osm_csv)
        except pd.errors.EmptyDataError:
            d = pd.DataFrame()   # ciudad sin generadores solares en OSM
        if len(d):
            frames.append(pd.DataFrame({"latitude": d.latitude, "longitude": d.longitude,
                                        "mwp": d.mwp, "source": "osm"}))
    gppd = os.path.join(PV_DIR, "gppd_solar.csv")
    if os.path.exists(gppd):
        g = pd.read_csv(gppd)
        g = g[g.capacity_mw > 0]
        frames.append(pd.DataFrame({"latitude": g.latitude, "longitude": g.longitude,
                                    "mwp": g.capacity_mw, "source": "gppd"}))
    if not frames:
        return None
    os.makedirs(MERGED_DIR, exist_ok=True)
    out = os.path.join(MERGED_DIR, f"{slug}.csv")
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out


def earthpv_city_layer(city: str, slug: str, query: str) -> tuple:
    """Recorta el atlas EarthPV nacional al entorno de la ciudad (±8 km).

    El atlas completo (p. ej. Sudáfrica: 2 600 celdas de 0.1°) se dispersa en
    retícula ANTES de recortar en pv_data.prepare_pv → pico de memoria enorme.
    Pre-recortarlo deja solo las celdas de la ciudad (grid2poster acepta una
    'capa derivada de EarthPV' por ruta en --earthpv).
    """
    pq, pais, tier = EARTHPV_BY_CITY[city]
    os.makedirs(MERGED_DIR, exist_ok=True)
    out = os.path.join(MERGED_DIR, f"{slug}_earthpv.parquet")
    if not os.path.exists(out):
        import geopandas as gpd
        import shapely
        from shapely.ops import unary_union
        from osm_data import get_country_boundary
        full = gpd.read_parquet(os.path.join(PV_DIR, pq))
        b = get_country_boundary(query, use_cache=True)
        mask = unary_union(b.to_crs("EPSG:3857").geometry).buffer(8000.0)
        shapely.prepare(mask)
        pts = full.to_crs("EPSG:3857").geometry.representative_point()
        keep = shapely.contains(mask, pts.values)
        full[keep].to_parquet(out)
        print(f"    EarthPV {pais}: {int(keep.sum())}/{len(full)} celdas cerca de {city}",
              flush=True)
    return out, pais, tier


def render_variant(cgp, city: str, slug: str, variant: str, query: str) -> tuple:
    out_dir = os.path.join(OUT_ROOT, variant)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{slug}.png")
    argv = ["create_grid_poster.py",
            "--country", query,
            "--display-country", city.upper(),
            "--single-query",
            "--theme", THEME,
            "--width", str(WIDTH_MM), "--height", str(HEIGHT_MM), "--dpi", str(DPI),
            "--output", out_png]
    detalle = ""
    if variant == "plantas":
        argv += ["--show-plants"]
        detalle = "OSM power=plant"
    else:  # solar
        if city in EARTHPV_BY_CITY:
            path, pais, tier = earthpv_city_layer(city, slug, query)
            argv += ["--earthpv", path]
            detalle = f"EarthPV {pais} ({tier})"
        else:
            merged = build_merged_solar_csv(slug)
            if merged:
                argv += ["--pv-layer", merged, "--pv-column", "mwp",
                         "--pv-label", "SOLAR PV (OSM+GPPD)"]
                detalle = "OSM+GPPD"
            else:
                detalle = "sin capa PV"
    t0 = time.time()
    rc = cgp.main(argv[1:])
    ok = (rc in (0, None)) and os.path.exists(out_png)
    size = os.path.getsize(out_png) if os.path.exists(out_png) else 0
    return ("OK" if ok else f"rc={rc}"), size, time.time() - t0, detalle


def main(variants: list[str], cities: list[str]) -> None:
    from render_city_maps import _install_overpass_retry, _geocode_query
    _install_overpass_retry()
    sys.path.insert(0, GRID2POSTER_DIR)
    import create_grid_poster as cgp
    from common import slugify

    total = len(variants) * len(cities)
    n = 0
    results = {}
    for variant in variants:
        for city in cities:
            n += 1
            slug = slugify(city)
            query = _geocode_query(city)
            try:
                st, size, dt, det = render_variant(cgp, city, slug, variant, query)
                results[(variant, city)] = st
                print(f"[{n:2d}/{total}] {variant:7s} {city:15s} -> {st} "
                      f"({size/1024:.0f} KB, {dt:.0f}s) {det}", flush=True)
            except Exception as e:  # noqa: BLE001
                results[(variant, city)] = "EXC"
                print(f"[{n:2d}/{total}] {variant:7s} {city:15s} -> EXCEPCIÓN "
                      f"{type(e).__name__}: {str(e)[:140]}", flush=True)
    nok = sum(1 for v in results.values() if v == "OK")
    print(f"\n=== {nok}/{total} mapas enriquecidos renderizados ===", flush=True)
    for k, v in results.items():
        if v != "OK":
            print(f"  FALLO {k}: {v}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    variants = ["plantas", "solar"]
    if args and args[0] in ("plantas", "solar", "ambas"):
        variants = ["plantas", "solar"] if args[0] == "ambas" else [args[0]]
        args = args[1:]
    import core.synthetic_data as sd
    main(variants, args if args else sd.list_cities())

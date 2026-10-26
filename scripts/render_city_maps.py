"""Renderiza el mapa REAL de la red de transmisión de cada ciudad con **grid2poster**
(Open Energy Transition): geocodifica el límite urbano vía Nominatim, descarga las
líneas `power=line` de OpenStreetMap vía Overpass y compone un póster con GeoPandas +
OSMnx + Matplotlib.

Guarda PNGs compactos en assets/city_maps/{slug}.png para incrustarlos en el dashboard
(imágenes estáticas ⇒ robusto en Streamlit Cloud, sin red en tiempo de ejecución).

Uso:
    python scripts/render_city_maps.py                 # las 16 ciudades
    python scripts/render_city_maps.py Bogotá Lima     # ciudades concretas
Requiere el checkout de grid2poster (ver GRID2POSTER_DIR) y sus dependencias.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- localizar grid2poster ---
GRID2POSTER_DIR = os.environ.get("GRID2POSTER_DIR", "/home/user/grid2poster")
if not os.path.isdir(GRID2POSTER_DIR):
    for cand in (os.path.join(ROOT, "grid2poster"), "/home/user/grid2poster"):
        if os.path.isdir(cand):
            GRID2POSTER_DIR = cand
            break

OUT_DIR = os.path.join(ROOT, "assets", "city_maps")
THEME = os.environ.get("G2P_THEME", "paper_grid")
WIDTH_MM, HEIGHT_MM, DPI = 200, 260, 100   # PNG compacto ≈ 787×1024 px

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def _install_overpass_retry():
    """Parchea requests.post para rotar espejos de Overpass con reintentos (502/429/504)."""
    import requests
    _orig = requests.post

    def patched(url, *a, **k):
        u = str(url)
        if "interpreter" not in u and "overpass" not in u:
            return _orig(url, *a, **k)
        # límite duro por petición: sin él, un mirror que gotea bytes resetea el
        # timeout de lectura y el fetch queda colgado indefinidamente
        k["timeout"] = tuple(int(x) for x in
                             os.environ.get("G2P_OVERPASS_TIMEOUT", "30,180").split(","))
        last = None
        for attempt in range(int(os.environ.get("G2P_OVERPASS_ATTEMPTS", "12"))):
            mirror = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
            try:
                r = _orig(mirror, *a, **k)
                if r.status_code in (429, 502, 503, 504):
                    last = f"HTTP {r.status_code} @ {mirror}"
                    time.sleep(4 + attempt * 3)
                    continue
                return r
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e} @ {mirror}"
                time.sleep(4 + attempt * 3)
        raise RuntimeError(f"Overpass agotó los reintentos: {last}")

    requests.post = patched
    try:
        import osmnx as ox
        ox.settings.overpass_url = OVERPASS_MIRRORS[0]
        ox.settings.requests_timeout = 120
    except Exception:  # noqa: BLE001
        pass


# Nombres que Nominatim no resuelve bien en español → consulta alternativa
GEOCODE_OVERRIDES = {
    "Johannesburgo": "City of Johannesburg Metropolitan Municipality",
    "Ciudad de México": "Mexico City, Mexico",
}


def _geocode_query(city: str) -> str:
    if city in GEOCODE_OVERRIDES:
        return GEOCODE_OVERRIDES[city]
    import core.synthetic_data as sd
    try:
        c = sd.get_city(city)
        country = c.get("country", "")
    except Exception:  # noqa: BLE001
        country = ""
    if not country or country.lower() == city.lower():
        return city
    # ayudar a Nominatim con el país (desambiguación)
    return f"{city}, {country}"


def main(cities: list[str]):
    _install_overpass_retry()
    sys.path.insert(0, GRID2POSTER_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    import core.synthetic_data as sd
    from common import slugify            # grid2poster
    import create_grid_poster as cgp

    all_cities = sd.list_cities()
    results = {}
    for i, city in enumerate(cities, 1):
        slug = slugify(city)
        out_png = os.path.join(OUT_DIR, f"{slug}.png")
        query = _geocode_query(city)
        argv = [
            "create_grid_poster.py",
            "--country", query,
            "--display-country", city.upper(),
            "--single-query",
            "--theme", THEME,
            "--width", str(WIDTH_MM), "--height", str(HEIGHT_MM), "--dpi", str(DPI),
            "--output", out_png,
        ]
        if os.environ.get("G2P_PLANTS", "1") == "1":
            argv += ["--show-plants", "--min-plant-capacity", "5"]
        t0 = time.time()
        try:
            rc = cgp.main(argv[1:])   # pasar explícito: main(argv=sys.argv[1:]) congela el default al importar
            ok = (rc in (0, None)) and os.path.exists(out_png)
            size = os.path.getsize(out_png) if os.path.exists(out_png) else 0
            results[city] = ("OK" if ok else f"rc={rc}", f"{size/1024:.0f} KB",
                             f"{time.time()-t0:.0f}s", query)
            print(f"[{i:2d}/{len(cities)}] {city:15s} -> {'OK' if ok else 'FALLO'} "
                  f"({size/1024:.0f} KB, {time.time()-t0:.0f}s) query='{query}'", flush=True)
        except Exception as e:  # noqa: BLE001
            results[city] = ("EXC", str(e)[:120], f"{time.time()-t0:.0f}s", query)
            print(f"[{i:2d}/{len(cities)}] {city:15s} -> EXCEPCIÓN {type(e).__name__}: {str(e)[:120]}",
                  flush=True)

    nok = sum(1 for v in results.values() if v[0] == "OK")
    print(f"\n=== {nok}/{len(cities)} mapas renderizados en {OUT_DIR} ===", flush=True)
    for c, v in results.items():
        if v[0] != "OK":
            print(f"  FALLO {c}: {v}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    import core.synthetic_data as sd
    targets = args if args else sd.list_cities()
    main(targets)

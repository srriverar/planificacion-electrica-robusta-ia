# ⚡ Planificación Robusta de Sistemas Eléctricos Urbanos con IA

Dashboard interactivo (Streamlit) que **materializa la metodología** de la tesis doctoral:

> **Modelo robusto de planificación de sistemas eléctricos para ciudades capitales e
> intermedias utilizando técnicas de inteligencia computacional.**

**Diseñadores**

| Rol | Nombre | Filiación |
|---|---|---|
| 🎓 Estudiante (proponente) | **Juan José Bernal Segura** | Doctorado en Ingeniería Eléctrica, Universidad Nacional de Colombia – Sede Bogotá |
| 👨‍🏫 Profesor (director) | **Sergio Raúl Rivera Rodríguez** | Departamento de Ingeniería Eléctrica, Universidad Nacional de Colombia – Sede Bogotá |

> ⚠️ **Todos los datos son 100 % sintéticos** y reproducibles (semillas fijas). Las
> poblaciones de las ciudades son estimaciones reales ≈2025 en el rango **6–9 millones
> de habitantes**; las series energéticas, de red, climáticas y de data centers son
> generadas para demostrar el modelo, no mediciones reales.

---

## 🧭 Qué hace cada pestaña

| Pestaña | Contenido | Herramienta | Objetivos |
|---|---|---|---|
| 🏠 **Inicio** | Tesis, diseñadores, arquitectura del modelo, tablero global de avance | — | OE-1…OE-4 |
| 🌍 **Ciudades y Datos** | 16 ciudades del mundo (6–9 M hab.), mapa, series horarias, K-Means y PCA, calidad de datos | scikit-learn | OE-1 |
| 🧠 **Predicción** | Red neuronal (MLP) + Gradient Boosting + ensemble, bandas de incertidumbre, interpretabilidad | scikit-learn | OE-2, OE-3 |
| ⚡ **PyPSA** | Expansión 2026–2035, despacho óptimo (LOPF/HiGHS), N-1, costos, CO₂ por escenario | **PyPSA** | OE-3, OE-4 |
| 🤖 **Grid2Op por ciudad** | Agente **DQN en NumPy** (refuerzo) operando la red **de cada ciudad** (crónicas desde su demanda) vs Do-Nothing y heurística | **Grid2Op** | OE-2 |
| 🖥️ **Vessim** | Co-simulación de hiperdatacenter: carga IA, PV, baterías, despacho *carbon-aware* | **Vessim** | OE-4, PI-1 |
| 🛡️ **Resiliencia** | Eventos extremos + N-1: plan determinista vs robusto-IA, barrido de severidad | PyPSA | OE-3, PI-4 |
| 🌱 **Sostenibilidad** | Metas CO₂/renovables/OPEX/eficiencia, radar urbano, comparación 16 ciudades | modelo anual | OE-4 |
| 📈 **Crecimiento y escenarios** | Proyección demográfica 2026–2045 × circunstancias × plan, con **conclusiones por ciudad** (déficit, CO₂, riesgo) | modelo anual | OE-1, OE-3, OE-4 |
| 🗺️ **Mapas de Red** | **Mapa real OSM** por ciudad en 3 subpestañas: 🔌 transmisión (km por tensión) + Grid2Op + crecimiento, 🏭 **centrales** (`--show-plants`: MW por fuente, top-10) y ☀️ **glow solar** (atlas **EarthPV** / **OSM+WRI GPPD**) | **grid2poster** (OSM), EarthPV, WRI GPPD | OE-1…OE-4 |
| 🎯 **Objetivos** | Registro de supuestos (S0–S10), metas, cronograma y despliegue | — | Todos |

**Cada pestaña muestra sus supuestos (📋) y un panel de avance de los objetivos de la tesis (🎯).**

### Objetivos específicos de la tesis
- **OE-1** Recopilar/analizar datos multidimensionales (caso Bogotá) con minería y aprendizaje no supervisado.
- **OE-2** Modelo híbrido de planeación con aprendizaje por refuerzo para optimización dinámica.
- **OE-3** Evaluar desempeño: precisión, eficiencia e integración de renovables.
- **OE-4** Evaluar impacto en sostenibilidad, eficiencia y costos con tecnologías futuras (VE, data centers).

---

## 🚀 Ejecutar en local

```bash
git clone https://github.com/<TU_USUARIO>/planificacion-electrica-robusta-ia.git
cd planificacion-electrica-robusta-ia
python -m venv .venv && source .venv/bin/activate     # opcional
pip install -r requirements.txt
streamlit run app.py
```

> **Compatibilidad:** `pypsa` está fijado en **1.2.4** porque `pypsa>=1.3` exige
> `pandas>=3.0`, mientras que `grid2op` (vía `pandapower`) requiere `pandas<3`. Esa
> combinación permite tener **PyPSA + Grid2Op + Vessim** en el mismo entorno.

### (Opcional) Re-entrenar el agente RL
El repo incluye pesos y evaluación precomputados (`models/dqn_case5.npz`,
`results/rl_eval.json`) para que la nube responda al instante. Para regenerarlos:

```bash
python scripts/train_dqn.py 250 500     # entrena el DQN (~10–20 min, CPU)
python scripts/eval_rl_agents.py        # evalúa 3 agentes en 20 crónicas estándar (~12 min)
python scripts/eval_rl_cities.py          # evalúa DQN vs Do-Nothing en las 16 ciudades (~28 min)
# Mapas reales (requiere clonar grid2poster y sus dependencias; los PNG ya vienen en el repo):
#   git clone --single-branch https://github.com/open-energy-transition/grid2poster
#   pip install osmnx geopandas pyarrow
#   python scripts/render_city_maps.py        # 🔌 mapa base OSM de las 16 ciudades
# Capas enriquecidas (🏭 centrales y ☀️ glow solar):
#   mkdir -p /home/user/pv_layers && cd /home/user/pv_layers
#   for c in morocco south_africa vietnam; do curl -O https://open-energy-transition.github.io/earthpv/assets/interactive/${c}_pv_evidence_atlas_capacity_by_cell.parquet; done
#   curl -o gppd.csv https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
#   python -c "import pandas as pd; g=pd.read_csv('gppd.csv',low_memory=False); s=g[g.primary_fuel=='Solar']; s[['country_long','name','capacity_mw','latitude','longitude','commissioning_year','owner']].to_csv('gppd_solar.csv',index=False)"
#   python scripts/fetch_osm_solar.py             # generadores solares OSM por ciudad
#   python scripts/render_city_maps_layers.py     # 🏭 plantas/ y ☀️ solar/ (16 ciudades ×2)
#   python scripts/postprocess_layers.py          # plants_stats.json + solar_stats.json
```

---

## ☁️ Desplegar en Streamlit Community Cloud
> ⚠️ **Python 3.12 obligatorio en el panel de Cloud**: Community Cloud ignora
> `runtime.txt` y su default (3.13/3.14) no tiene wheels para algunas versiones
> fijadas aquí → el build queda colgado en "Your app is in the oven".
> En share.streamlit.io: **Manage app → Settings → Advanced settings → Python
> version → 3.12** y luego *Reboot* / redeploy.

1. Sube el repo a GitHub (ver abajo).
2. Entra a <https://share.streamlit.io> con tu cuenta de GitHub.
3. **New app** → selecciona el repositorio, rama `main`, archivo principal `app.py`.
4. **Deploy**. Streamlit instala `requirements.txt` automáticamente.

> **Grid2Op en la nube:** el ambiente `rte_case5_example` viene **incluido en el
> paquete**, así que la app funciona sin descargas adicionales. Ambientes L2RPN mayores
> (`l2rpn_case14_sandbox`, etc.) requieren descarga y no se usan aquí.
>
> **Rendimiento:** la pestaña 🧠 entrena modelos al primer uso (~20–40 s por ciudad) y
> ⚡ resuelve LPs (1–3 s cada uno). Todo se **cachea**. En el plan gratuito de Streamlit
> Cloud (recursos limitadas) prefiere los valores por defecto; los botones de cálculos
> pesados (trayectorias 30 LP, N-1, barridos, 16 ciudades) son opcionales bajo demanda.

---

## 📤 Subir a GitHub

```bash
cd planificacion-electrica-robusta-ia
git init
git add .
git commit -m "Dashboard tesis: planificación robusta de sistemas eléctricos urbanos con IA (PyPSA + Grid2Op + Vessim)"
git branch -M main
git remote add origin https://github.com/<TU_USUARIO>/planificacion-electrica-robusta-ia.git
git push -u origin main
```

Los artefactos del demo (`models/dqn_case5.npz`, `results/rl_eval.json`, `results/rl_city_eval.json`,
`results/dqn_train.json`) **están versionados** a propósito para que la nube muestre
resultados sin re-entrenar.

---

## 🗂️ Estructura

```
app.py                     # Entrada Streamlit: sidebar + 10 pestañas
core/
  config.py                # metadatos de la tesis, diseñadores, objetivos, supuestos (S0–S9)
  synthetic_data.py        # catálogo de 16 ciudades + generadores de series sintéticas
  forecasting.py           # minería (K-Means/PCA) + predicción NN/ensemble + interpretabilidad
  pypsa_planning.py        # red sintética, despacho, expansión, N-1, eventos extremos
  grid2op_rl.py            # capa RL: ambiente, agentes, entrenamiento/evaluación
  city_env.py              # Grid2Op POR CIUDAD: crónicas derivadas de la demanda sintética
  dqn.py                   # DQN minimalista en NumPy (MLP + replay + target net)
  vessim_dc.py             # co-simulación de data centers con Vessim
  sustainability.py        # modelo anual rápido de sostenibilidad (OE-4)
  population_scenarios.py  # escenarios de crecimiento poblacional + conclusiones (OE-1/3/4)
  ui_helpers.py            # paneles de supuestos/objetivos, KPIs, colores
tabs/                      # una función render() por pestaña (t1..t11)
assets/city_maps/          # mapas OSM pre-renderizados con grid2poster (PNG por ciudad)
                           #   + plantas/ y solar/ (capas 🏭 y ☀️), coverage.json,
                           #   plants_stats.json, solar_stats.json
data/osm_solar/            # generadores solares OSM por ciudad (build; sin versionar)
scripts/                   # train_dqn.py, eval_rl_agents.py, eval_rl_cities.py,
                           # render_city_maps.py, render_city_maps_layers.py,
                           # fetch_osm_solar.py, postprocess_coverage.py, postprocess_layers.py
models/ results/           # artefactos precomputados del demo
docs/SUPUESTOS.md          # registro consolidado de supuestos
```

## 🧰 Stack
PyPSA · Grid2Op (L2RPN) · Vessim · **grid2poster** (mapas OSM) · scikit-learn · Plotly · Streamlit · HiGHS.
Fuentes geoespaciales: OpenStreetMap (ODbL) · Nominatim · Overpass · **EarthPV** (Open Energy Transition) · **WRI Global Power Plant Database** (CC BY 4.0).

---

*Diseñado por Juan José Bernal Segura (estudiante) y Sergio Raúl Rivera Rodríguez
(profesor director) — Universidad Nacional de Colombia, Sede Bogotá. Datos sintéticos
para demostración metodológica.*

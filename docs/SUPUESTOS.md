# 📋 Registro consolidado de supuestos del dashboard (S0–S10)

> Todos los datos son **sintéticos y reproducibles** (semillas fijas por ciudad).
> Las poblaciones son estimaciones reales ≈2025 en el rango 6–9 M hab. Este registro
> replica lo que cada pestaña muestra en su panel "Supuestos".

## S0 · Globales (🏠 Inicio)
- **S0.** Datos 100 % sintéticos y reproducibles (semillas fijas); poblaciones ≈ reales 2025 (6–9 M hab.).
- **S0.1** El dashboard es una demo de la metodología: cada pestaña materializa una etapa del modelo robusto (datos → predicción NN → planificación PyPSA → operación RL Grid2Op → data centers Vessim → resiliencia → sostenibilidad).

## S1 · Ciudades y Datos (🌍, OE-1)
- **S1.1** Catálogo de 16 ciudades del mundo con población actual entre 6 y 9 M (ciudad o área metropolitana según se indique).
- **S1.2** Demanda = población × consumo per cápita plausible, con formas diarias/semanales/estacionales por sector, clima (latitud), tendencia, VE y data centers.
- **S1.3** Series horarias de 2 años (caso base) y mensuales de 10 años (tendencias); ruido AR(1).
- **S1.4** Minería: K-Means (k=4) sobre perfiles diarios normalizados y PCA sobre indicadores; meta ≥ 5 patrones clave.

## S2 · Predicción (🧠, OE-2/OE-3)
- **S2.1** Partición temporal 70/15/15 sin barajar; horizonte directo h ∈ {1, 6, 12, 24} h.
- **S2.2** Features: hora/día/mes cíclicos, temperatura, radiación, lags 24/48/168 h, medias móviles, VE y data centers.
- **S2.3** "Red neuronal" = MLPRegressor (2 capas ReLU) comparada con persistencia, naive estacional, Gradient Boosting y ensemble apilado (Ridge).
- **S2.4** Bandas de incertidumbre: cuantiles 5–95 % del Gradient Boosting.
- **S2.5** Metas de la propuesta: MAPE ≤ 5 % y mejora del ensemble ≥ 15 % vs mejor individual.

## S3 · Planificación PyPSA (⚡, OE-3/OE-4)
- **S3.1** Red sintética de 7 buses por ciudad (interconexión 220 kV, centro 220 kV, 5 zonas 110 kV); flujo DC (LOPF) sin unit commitment.
- **S3.2** Parque coherente con la matriz de la ciudad + batería 4 h + racionamiento con VOLL = 500 USD/MWh.
- **S3.3** Despacho sobre 96 snapshots representativos (4 estaciones × 24 h) ponderados; expansión 2026→2035 por escenario.
- **S3.4** Costos/CAPEX y factores de capacidad sintéticos; precio de CO₂ configurable; sin modelar redes de distribución.

## S4 · Grid2Op / RL (🤖, OE-2)
- **S4.1** Malla física `rte_case5_example` (5 subestaciones, 3 cargas, 2 generadores, 8 líneas) como banco de pruebas estándar de L2RPN. La red específica por ciudad derivada de PyPSA es trabajo futuro del OE-2; lo que cambia **por ciudad** es la crónica temporal que alimenta la red.
- **S4.2** DQN propio en NumPy: MLP 64–64, replay 20 000, ε-greedy 1.0→0.05, target network, 25 acciones discretas con máscara de legalidad. Pesos únicos transferibles entre ciudades (misma topología).
- **S4.3** Pesos pre-entrenados y evaluaciones precomputadas en el repo (`results/rl_eval.json` global, `results/rl_city_eval.json` por ciudad) para respuesta instantánea en la nube.
- **S4.4** **Grid2Op POR CIUDAD** (`core/city_env.py`): cada una de las 16 ciudades tiene un ambiente propio cuyas crónicas (`load_p/load_q/prod_p/prod_v` a 5 min, 5 variantes-semana) se re-forman desde su curva de demanda sintética —factor de pico, mezcla renovable y crecimiento—. Las magnitudes se mantienen en el rango de la malla (≈MW, pico total 26–29,5 MW) para que los pesos entrenados transfieran; la dificultad relativa entre ciudades emerge de la forma de la demanda y la variabilidad renovable.
- **S4.5** Recompensa LinesCapacityReward; comparación contra Do-Nothing y PowerLineSwitch. Resultado: el DQN sobrevive las 5 crónicas en las 16 ciudades (80/80); Do-Nothing colapsa según la dificultad de cada ciudad.

## S5 · Data centers Vessim (🖥️, OE-4/PI-1)
- **S5.1** Carga IT con forma diurna (inferencia diurna + entrenamiento nocturno) × PUE configurable.
- **S5.2** PV y baterías dimensionados como fracción del pipeline del data center; señales horarias de intensidad de carbono y precio de la ciudad.
- **S5.3** Políticas: sin gestión, batería por defecto y *carbon-aware* propia (carga con red limpia/barata).
- **S5.4** Co-simulación nativa a 15 min sobre 7–30 días; KPIs: autosuficiencia, importación, emisiones, costo, pico.

## S6 · Resiliencia (🛡️, OE-3/PI-4)
- **S6.1** Eventos sintéticos: pico de demanda, ola de calor, tormenta, falla de generación, surgimiento de data centers.
- **S6.2** Comparación determinista (BAU) vs robusto-IA sobre el mismo despacho con racionamiento explícito.
- **S6.3** N-1 con re-despacho completo; violación si ENS > 0 o carga de línea > 100 %.
- **S6.4** Métricas: ENS (MWh), sobrecarga máxima, costo, CO₂; barrido de severidad 0–30 %.

## S7 · Sostenibilidad (🌱, OE-4)
- **S7.1** Modelo anual (capacidad × factor de capacidad) consistente con las reglas de expansión PyPSA.
- **S7.2** Metas: +25 pp renovables, −20 % CO₂, −15 % OPEX, +30 % eficiencia (2026→2031 vs BAU).
- **S7.3** Radar urbano Demanda/Resiliencia/Sostenibilidad con índices 0–100.

## S8 · Objetivos y despliegue (🎯)
- **S8.1** El avance de cada objetivo se calcula con evidencias del propio demo sobre datos sintéticos; la validación con datos reales de Bogotá (XM, UPME, IDEAM, distribuidores) es la fase siguiente del cronograma.

## S9 · Crecimiento poblacional y escenarios (📈, OE-1/OE-3/OE-4)
- **S9.1** Escenarios demográficos por ciudad (2026→2045): Estancamiento (0,2× la tendencia), Tendencia ONU (1×, proxy ONU/Hábitat), Acelerado (1,5× con tope 6 %/año) y Boom tecnológico (1,25× + data centers 2,6× + VE 2×). Se aplica un **tope de plausibilidad** para evitar explosiones demográficas irreales.
- **S9.2** Circunstancias activables: 🚶 ola migratoria (+0,5 %/año desde 2028), 🌡️ olas de calor crónicas (refrigeración +2 %/año, pico +5 %), 🖥️ boom de data centers (carga IA ×2), 🔌 electrificación agresiva (consumo per cápita +1,2 %/año), 📉 recesión (−0,4 %/año), 🏜️ sequía renovable (−15 % hidro/solar/eólica).
- **S9.3** Plan de respuesta (oferta): BAU tendencial, Transición moderada o Robusto-IA (reglas de expansión de PyPSA) con **tope de saturación de 4× la capacidad base** para no divergir en horizontes largos.
- **S9.4** Modelo anual rápido: población → demanda (per cápita × ahorro de eficiencia + data centers) → oferta (capacidad × factor de capacidad) → margen de capacidad de pico, CO₂ efectivo (oferta + déficit cubierto con gas) y % renovable. Coherente con las reglas de PyPSA; el despacho horario fino vive en ⚡/🛡️.
- **S9.5** Riesgo por margen de capacidad: 🔴 crítico (<0 %), 🟠 alto (<15 %), 🟡 vigilar (<30 %), 🟢 estable (≥30 %). La capacidad adicional requerida (GW) es la **brecha de adecuación de pico** (no la de energía). Las conclusiones por ciudad y globales se generan automáticamente (lenguaje natural basado en reglas).

## S10 · Mapas reales de red por ciudad (🗺️, grid2poster)
- **S10.1** El mapa de cada ciudad se renderiza con **grid2poster** (Open Energy Transition): límite urbano geocodificado vía Nominatim + líneas `power=line` reales de OpenStreetMap vía Overpass, dibujadas con GeoPandas + OSMnx + Matplotlib (tema `paper_grid`). La pestaña tiene tres subpestañas: 🔌 red base, 🏭 centrales (`--show-plants`) y ☀️ glow solar (`--earthpv`/`--pv-layer`).
- **S10.2** La cobertura de la red en OSM varía por región (mejor en Europa/Norteamérica, más escasa en África/Asia); los km por nivel de tensión del pie del mapa (y `coverage.json`) son los mapeados por la comunidad (MapYourGrid).
- **S10.3** Los mapas se **pre-renderizan offline** (`scripts/render_city_maps.py`) y se versionan como PNG en `assets/city_maps/`, de modo que el dashboard no depende de la red en tiempo de ejecución (robusto en Streamlit Cloud). grid2poster es dependencia de *build*, no de runtime.
- **S10.4** Junto al mapa se anclan el episodio Grid2Op del ambiente propio de la ciudad (🤖) y su conclusión de crecimiento (📈): es la misma evidencia, georreferenciada sobre el territorio real.
- **S10.5** Subpestaña 🏭: las centrales son las mapeadas en OSM (`power=plant`), con marcadores dimensionados por `plant:output:electricity` y coloreados por `plant:source`; la cobertura depende del mapeo comunitario de cada región (hay ciudades con pocas o ninguna central mapeada).
- **S10.6** Subpestaña ☀️: la capa solar sigue una **cascada de fuentes abiertas** — (1) atlas **EarthPV** por celdas 0.1° (`mwp_best`, fase alfa) para Marruecos/Sudáfrica/Vietnam (Casablanca, Johannesburgo, Hanói); (2) generadores **OSM** `power=generator` + `generator:source=solar`; (3) subconjunto solar de **WRI Global Power Plant Database** (10 665 plantas, 188 GW, 167 países, foto ≈2018-2020, CC BY 4.0).
- **S10.7** Un generador solar de OSM sin etiqueta de potencia aporta **20 kWp asumidos** (instalación residencial típica) para el glow; las centrales sin capacidad conocida se cuentan pero no suman MW.
- **S10.8** Puede haber **solapamiento OSM↔GPPD** (la misma planta en ambas fuentes): el glow es ilustrativo de la densidad solar, no una contabilidad oficial de capacidad.
- **S10.9** Ciudades sin geodato solar abierto en ninguna fuente (p. ej. Bogotá, Teherán, Lima) se renderizan **sin glow** y se declara explícitamente: es disponibilidad honesta de datos, no un fallo.
- **S10.10** Créditos de datos: © OpenStreetMap contributors (ODbL), MapYourGrid, Open Energy Transition (grid2poster), EarthPV, WRI Global Power Plant Database (CC BY 4.0).

## 🔎 Fuentes reales para la siguiente fase
- **Bogotá:** XM S.A. E.S.P. (operador del SIN), UPME, CREG, IDEAM (clima), Enel X / Grupo Energía Bogotá.
- **Europa:** ENTSO-E Transparency, Open Power System Data (OPSD).
- **Renovables global:** Solcast / NREL NSRDB (solar), NASA POWER (viento), WattTime / electricityMap (carbono).
- **Data centers:** IEA Data Centres & Networks, reportes de operadores (PUE, cargas IT).

# -*- coding: utf-8 -*-
"""
Metadatos de la tesis, diseñadores, objetivos y registro central de
supuestos por pestaña.

Tesis: "Modelo robusto de planificación de sistemas eléctricos para ciudades
capitales e intermedias utilizando técnicas de inteligencia computacional"
Doctorado en Ingeniería Eléctrica — Universidad Nacional de Colombia, Sede Bogotá.
"""

APP_TITLE = "Planificación Robusta de Sistemas Eléctricos Urbanos con IA"
APP_SHORT = "SISEL-IA · Dashboard de tesis doctoral"
APP_ICON = "⚡"

DISENADORES = {
    "estudiante": {
        "nombre": "Juan José Bernal Segura",
        "rol": "Estudiante de Doctorado en Ingeniería Eléctrica",
        "institucion": "Universidad Nacional de Colombia – Sede Bogotá / Escuela Tecnológica Instituto Técnico Central",
    },
    "profesor": {
        "nombre": "Sergio Raúl Rivera Rodríguez",
        "rol": "Director de tesis · Profesor, Departamento de Ingeniería Eléctrica",
        "institucion": "Universidad Nacional de Colombia – Sede Bogotá",
    },
}

TITULO_TESIS = (
    "Modelo robusto de planificación de sistemas eléctricos para ciudades "
    "capitales e intermedias utilizando técnicas de inteligencia computacional"
)

HIPOTESIS = (
    "Un modelo de planeación de sistemas eléctricos de potencia mediante el uso de "
    "redes neuronales permite mejorar la precisión en la predicción de la demanda "
    "energética, optimizando la distribución de recursos y aumentando la resiliencia "
    "del sistema ante variaciones causadas por implementación de nuevas tecnologías "
    "(transporte eléctrico, data centers, comunicaciones), aumento demográfico o "
    "contingencias causadas por desastres naturales."
)

OBJETIVO_GENERAL = (
    "Desarrollar un modelo robusto e integrado de planificación de sistemas "
    "eléctricos de potencia para ciudades capitales e intermedias mediante técnicas "
    "avanzadas de aprendizaje automático e inteligencia computacional."
)

OBJETIVOS_ESPECIFICOS = {
    "OE-1": (
        "Recopilar y analizar datos multidimensionales sobre demanda eléctrica, "
        "infraestructura existente y variables externas que influyan en la planeación "
        "(caso de estudio: Bogotá), usando minería de datos y aprendizaje no "
        "supervisado para identificar patrones y relaciones complejas."
    ),
    "OE-2": (
        "Desarrollar un modelo híbrido de planeación de sistemas eléctricos que "
        "integre algoritmos de aprendizaje por refuerzo para la optimización dinámica "
        "de recursos, según las necesidades de ciudades capitales e intermedias."
    ),
    "OE-3": (
        "Evaluar el desempeño del modelo en términos de precisión, eficiencia "
        "energética y capacidad de integración de fuentes renovables."
    ),
    "OE-4": (
        "Evaluar el impacto del modelo en sostenibilidad, eficiencia energética y "
        "costos de operación para ciudades capitales e intermedias, considerando la "
        "inserción de tecnologías futuras (vehículos eléctricos, hiperdatacenters)."
    ),
}

PREGUNTAS_INVESTIGACION = [
    ("PI-1", "¿Cómo influye el aumento de carga eléctrica por nuevas tecnologías (VE, data centers) en los sistemas de potencia de ciudades capitales e intermedias?"),
    ("PI-2", "¿Cómo pueden las redes neuronales mejorar la planificación de sistemas eléctricos en ciudades capitales e intermedias?"),
    ("PI-3", "¿Qué impacto tienen los modelos de redes neuronales en la reducción de la incertidumbre en la planificación?"),
    ("PI-4", "¿De qué manera la robustez del modelo garantiza la estabilidad ante eventos extremos (picos de demanda, fallos de red, fenómenos climáticos)?"),
    ("PI-5", "¿Cómo pueden las redes neuronales facilitar la integración de fuentes renovables en la planificación urbana?"),
    ("PI-6", "¿Qué desafíos existen al implementar modelos basados en redes neuronales en ciudades intermedias con infraestructura menos desarrollada?"),
]

# Metas cuantitativas tomadas de las actividades de la propuesta (comentario SR4)
METAS = {
    "mape_max_pct": 5.0,          # error máximo de predicción diaria (%)
    "eficiencia_pct": 90.0,       # eficiencia en distribución de recursos (%)
    "mejora_ensemble_pct": 15.0,  # mejora del ensemble vs mejor modelo individual
    "renovables_extra_pct": 25.0, # aumento de participación renovable
    "co2_menos_pct": 20.0,        # reducción de CO2 en generación
    "eficiencia_global_pct": 30.0,# aumento de eficiencia energética global
    "opex_menos_pct": 15.0,       # reducción de costos operativos a 5 años
    "explicabilidad_pct": 90.0,   # predicciones críticas con explicación clara
    "adaptacion_h": 1.0,          # respuesta a cambios imprevistos (< 1 h)
}

HERRAMIENTAS = [
    ("PyPSA", "1.2.x", "Planificación de expansión y despacho óptimo (OPF/LOPF) de la red sintética de cada ciudad. Soporta OE-3 y OE-4."),
    ("Grid2Op (L2RPN)", "1.12.x", "Ambiente de aprendizaje por refuerzo para operación dinámica de la red (agente híbrido). Soporta OE-2."),
    ("Vessim", "0.16.x", "Co-simulación de microredes de data centers: demanda computacional, PV on-site, baterías y red pública. Soporta OE-4 y PI-1."),
    ("scikit-learn", "1.6.x", "Redes neuronales (MLP), ensembles (Gradient Boosting), clustering (K-Means) y PCA para minería de datos. Soporta OE-1 y OE-2."),
    ("Streamlit", "1.66.x", "Dashboard interactivo desplegable en Streamlit Community Cloud desde GitHub."),
]

# ---------------------------------------------------------------------------
# Registro de supuestos y objetivos por pestaña
# ---------------------------------------------------------------------------
TAB_INFO = {
    "inicio": {
        "supuestos": [
            "S0. Todos los datos del dashboard son 100 % sintéticos y reproducibles (semillas fijas); las poblaciones de las ciudades son valores aproximados reales 2025 en el rango 6–9 millones.",
            "S0.1 El dashboard es una demo de la metodología de la tesis: cada pestaña materializa una etapa del modelo robusto (datos → predicción NN → planificación PyPSA → operación RL Grid2Op → data centers Vessim → resiliencia → sostenibilidad).",
        ],
        "objetivos": [],
    },
    "ciudades": {
        "supuestos": [
            "S1.1 Catálogo de 16 ciudades del mundo cuya población actual (ciudad o área metropolitana, según se indique) está entre 6 y 9 millones de habitantes.",
            "S1.2 Demanda eléctrica sintética construida como: población × consumo per cápita eléctrico plausible, con formas diarias/semanales/estacionales por sector (residencial, comercial, industrial), clima local (latitud), tendencia de crecimiento, carga de vehículos eléctricos y de data centers.",
            "S1.3 Series horarias de 2 años para el caso base y agregados mensuales de 10 años para tendencias; ruido AR(1) para realismo.",
            "S1.4 Minería de datos: K-Means (k=4) sobre perfiles diarios normalizados y PCA sobre indicadores urbanos; número de patrones clave meta ≥ 5.",
        ],
        "objetivos": ["OE-1"],
    },
    "prediccion": {
        "supuestos": [
            "S2.1 Entrenamiento con 80 % histórico / validación 10 % / prueba 10 % en serie horaria (sin barajar), horizonte directo multi-paso h ∈ {1, 6, 12, 24} h.",
            "S2.2 Features: hora/día/mes (cíclicos), temperatura, radiación,lags 24/168 h, medias móviles, carga VE y data center.",
            "S2.3 'Red neuronal' = MLPRegressor de scikit-learn (2 capas, ReLU); se compara con persistencia, estacional-naive, Gradient Boosting y un ensemble apilado (Ridge meta-learner).",
            "S2.4 Bandas de incertidumbre por cuantiles del Gradient Boosting (5–95 %) y desviación de residuos del ensemble.",
            "S2.5 Meta de la propuesta: MAPE ≤ 5 % en predicción diaria y mejora ≥ 15 % del ensemble frente al mejor modelo individual.",
        ],
        "objetivos": ["OE-2", "OE-3"],
    },
    "pypsa": {
        "supuestos": [
            "S3.1 Red sintética por ciudad: 7 buses (interconexión 220 kV, centro 220 kV y 5 zonas 110 kV), líneas en anillo + radiales, flujo DC (LOPF) sin unit commitment.",
            "S3.2 Parque generador coherente con la matriz de cada ciudad (hidro, gas, carbón, nuclear, solar, eólico, importaciones) + batería 4 h + generador de racionamiento (valor de pérdida de carga VOLL = 500 USD/MWh).",
            "S3.3 Despacho sobre 96 snapshots representativos (4 estaciones × 24 h) ponderados; expansión año a año 2026→2035 con reglas por escenario (BAU, Transición moderada, Robusto-IA).",
            "S3.4 Costos y factores de capacidad sintéticos por tecnología; precio de CO2 configurable; inversiones CAPEX simplificadas (sin redes de distribución).",
        ],
        "objetivos": ["OE-3", "OE-4"],
    },
    "grid2op": {
        "supuestos": [
            "S4.1 Malla física `rte_case5_example` (5 subestaciones, 3 cargas, 2 generadores, 8 líneas) como banco de pruebas estándar de L2RPN; la red específica por ciudad derivada de PyPSA es la siguiente fase del OE-2. Lo que cambia POR CIUDAD es la crónica temporal que alimenta la red.",
            "S4.2 Agente RL propio: DQN implementado en NumPy (MLP 64-64, replay 20 000, ε-greedy 1.0→0.05, target network), espacio de 25 acciones discretas (do-nothing, reconexión de líneas, cambios topológicos) con máscara de acciones legales.",
            "S4.3 Pesos del DQN pre-entrenados offline (scripts/train_dqn.py) y evaluación precomputada (results/rl_eval.json) para que el dashboard responda en segundos; evaluación en vivo opcional.",
            "S4.4 Grid2Op POR CIUDAD (core/city_env.py): cada una de las 16 ciudades tiene un ambiente propio cuyas crónicas (load_p/load_q/prod_p/prod_v a 5 min) se re-forman desde su curva de demanda sintética — factor de pico, mezcla renovable y crecimiento — manteniendo las magnitudes en el rango de la malla (≈MW) para que los pesos entrenados transfieran. 5 crónicas-semana por ciudad.",
            "S4.5 Recompensa LinesCapacityReward (pasos sobrevividos sin congestión ilegal); se compara contra Do-Nothing y la heurística PowerLineSwitch. Evaluación por ciudad precomputada en results/rl_city_eval.json (scripts/eval_rl_cities.py) y episodios en vivo opcionales.",
        ],
        "objetivos": ["OE-2"],
    },
    "vessim": {
        "supuestos": [
            "S5.1 Data center sintético por ciudad: carga IT con forma diurna de workloads de IA (entrenamiento nocturno + inferencia diurna), PUE configurable (1.15–1.6); la potencia total IT+cooling se inyecta como Consumer de Vessim.",
            "S5.2 PV on-site y baterías dimensionadas en proporción al pipeline de MW del data center; conexión a red pública con señal de intensidad de carbono y precio horarios de la ciudad.",
            "S5.3 Políticas comparadas: (a) sin gestión, (b) batería con DefaultDispatchPolicy, (c) despacho carbon-aware (controller que fija charge_power según intensidad de carbono y precio).",
            "S5.4 Co-simulación nativa de Vessim a paso de 15 min sobre 30 días; KPIs: autosuficiencia %, energía importada, emisiones, costo y recorte de pico.",
        ],
        "objetivos": ["OE-4"],
    },
    "resiliencia": {
        "supuestos": [
            "S6.1 Eventos extremos sintéticos: pico de demanda (+%), ola de calor (derating solar/térmico + carga de refrigeración), tormenta (salida de líneas + caída eólica), falla de generación y surgimiento de data centers.",
            "S6.2 Comparación 'planificación determinista tradicional' (margen N-0, sin almacenamiento) vs 'modelo robusto-IA' (margen de reserva, baterías, respuesta de demanda) sobre el mismo despacho PyPSA con racionamiento explícito.",
            "S6.3 Análisis N-1 con flujo de potencia lineal sobre la hora pico: violaciones = |flujo| > s_nom tras contingencia.",
            "S6.4 Métricas: energía no suministrada (ENS MWh), sobrecarga máxima, costo y CO2 del evento; barrido de severidad 0–30 %.",
        ],
        "objetivos": ["OE-3"],
    },
    "sostenibilidad": {
        "supuestos": [
            "S7.1 Modelo anual simplificado (capacidad × factor de capacidad por ciudad/año) para trayectorias 2026–2035 de los 3 escenarios de planificación; consistente con los LP de la pestaña PyPSA.",
            "S7.2 Metas de la propuesta: +25 % renovables, −20 % CO2, −15 % OPEX, +30 % eficiencia global a 5 años.",
            "S7.3 Radar urbano sobre las tres dimensiones de la propuesta: Demanda, Resiliencia y Sostenibilidad (índices 0–100 construidos con los indicadores sintéticos).",
        ],
        "objetivos": ["OE-4"],
    },
    "poblacion": {
        "supuestos": [
            "S9.1 Escenarios demográficos por ciudad (2026→2045): Estancamiento (0,2× la tendencia), Tendencia ONU (1×, proxy de proyecciones ONU/Hábitat), Acelerado (1,5× con tope 6 %/año) y Boom tecnológico (1,25× + data centers 2,6× + VE 2×). Se aplica un tope de plausibilidad para evitar explosiones demográficas irreales.",
            "S9.2 Circunstancias activables: 🚶 ola migratoria (+0,5 %/año desde 2028), 🌡️ olas de calor crónicas (refrigeración +2 %/año, pico +5 %), 🖥️ boom de data centers (carga IA ×2), 🔌 electrificación agresiva (consumo per cápita +1,2 %/año), 📉 recesión (−0,4 %/año), 🏜️ sequía renovable (−15 % hidro/solar/eólica).",
            "S9.3 Plan de respuesta (oferta): BAU tendencial, Transición moderada o Robusto-IA (tesis), usando las reglas de expansión de PyPSA con un tope de saturación de 4× la capacidad base para no divergir en horizontes largos.",
            "S9.4 Modelo anual rápido: población → demanda (per cápita × ahorro de eficiencia + data centers) → oferta (capacidad × factor de capacidad) → margen de capacidad de pico, CO₂ efectivo (oferta + déficit cubierto con gas) y % renovable. Coherente con las reglas de PyPSA; el despacho horario fino vive en las pestañas ⚡ y 🛡️.",
            "S9.5 Riesgo por margen de capacidad: 🔴 crítico (<0 %), 🟠 alto (<15 %), 🟡 vigilar (<30 %), 🟢 estable (≥30 %). La capacidad adicional requerida (GW) es la brecha de adecuación de pico, no la de energía.",
        ],
        "objetivos": ["OE-1", "OE-3", "OE-4"],
    },
    "mapas": {
        "supuestos": [
            "S10.1 El mapa de cada ciudad se renderiza con grid2poster (Open Energy Transition): límite urbano geocodificado vía Nominatim y líneas `power=line` reales de OpenStreetMap descargadas vía Overpass, dibujadas con GeoPandas + OSMnx + Matplotlib (tema paper_grid).",
            "S10.2 La cobertura de la red en OSM varía por región (mejor en Europa/Norteamérica); los km por nivel de tensión que aparecen en el pie del mapa son los mapeados por la comunidad (MapYourGrid).",
            "S10.3 Los mapas se pre-renderizan offline (scripts/render_city_maps.py) y se versionan como PNG en assets/city_maps/ para que el dashboard no dependa de la red en tiempo de ejecución (robusto en Streamlit Cloud).",
            "S10.4 Junto al mapa se anclan el episodio Grid2Op del ambiente propio de la ciudad (🤖) y su conclusión de crecimiento poblacional (📈): es la misma evidencia, georreferenciada.",
            "S10.5 Subpestaña 🏭: las centrales son las mapeadas en OSM (`power=plant`), marcadores dimensionados por `plant:output:electricity` y coloreados por `plant:source`; la cobertura depende del mapeo comunitario de cada región (hay ciudades con pocas o ninguna central mapeada).",
            "S10.6 Subpestaña ☀️: la capa solar sigue una cascada de fuentes abiertas — (1) atlas EarthPV por celdas 0.1° (`mwp_best`, fase alfa) para Marruecos/Sudáfrica/Vietnam (Casablanca, Johannesburgo, Hanói); (2) generadores OSM `power=generator` + `generator:source=solar`; (3) subconjunto solar de WRI Global Power Plant Database (10 665 plantas, 188 GW, 167 países, foto ≈2018-2020, CC BY 4.0).",
            "S10.7 Un generador solar de OSM sin etiqueta de potencia aporta 20 kWp asumidos (instalación residencial típica) para el glow; las centrales sin capacidad conocida se cuentan pero no suman MW.",
            "S10.8 Puede haber solapamiento OSM↔GPPD (la misma planta en ambas fuentes): el glow es ilustrativo de la densidad solar, no una contabilidad oficial de capacidad.",
            "S10.9 Ciudades sin geodato solar abierto en ninguna fuente (p. ej. Bogotá, Teherán, Lima) se renderizan sin glow y se declara explícitamente: es disponibilidad honesta de datos, no un fallo.",
        ],
        "objetivos": ["OE-1", "OE-2", "OE-3", "OE-4"],
    },
    "objetivos": {
        "supuestos": [
            "S8.1 El estado de avance de cada objetivo se calcula con las evidencias del propio demo (KPIs vivos), no con datos reales de Bogotá; la recolección real (XM, UPME, IDEAM, operadores) es la fase siguiente del cronograma.",
        ],
        "objetivos": ["OE-1", "OE-2", "OE-3", "OE-4"],
    },
}

FUENTES_REALES_FUTURAS = [
    ("Bogotá (caso de estudio)", "XM S.A. E.S.P. (operador SIN), UPME, CREG, IDEAM (clima), Enel X / Grupo Energía Bogotá (distribución)"),
    ("Europa", "ENTSO-E Transparency Platform, Open Power System Data (OPSD)"),
    ("Global renovables", "Solcast / NREL NSRDB (solar), NASA POWER (viento), WattTime / electricityMap (intensidad de carbono)"),
    ("Data centers", "IEA Data Centres & Networks, informes de operadores (PUE, cargas IT)"),
]

FOOTER = (
    "Diseñado por: **Juan José Bernal Segura** (estudiante doctoral) y "
    "**Sergio Raúl Rivera Rodríguez** (profesor director) — Universidad Nacional "
    "de Colombia, Sede Bogotá. Datos 100 % sintéticos para demostración."
)

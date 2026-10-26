# -*- coding: utf-8 -*-
"""
Capa de aprendizaje por refuerzo (Grid2Op / L2RPN) para la operación dinámica
de la red — modelo híbrido del OE-2.

Ambiente: `rte_case5_example` (5 subestaciones, 8 líneas, crónicas semanales a
5 min) como proxy didáctico de la red urbana sintética. Agentes:
  - DoNothing (línea base)
  - Heurística PowerflowSwitch (conmutación topológica ante congestión)
  - DQN NumPy propio (core/dqn.py) pre-entrenado offline

La dificultad por ciudad se emula con la selección de crónica + semilla
(etiqueta de severidad según crecimiento y factor pico de la ciudad).
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from . import synthetic_data as sd
from .dqn import NumpyDQN

ENV_NAME = "rte_case5_example"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(ROOT, "models", "dqn_case5.npz")
EVAL_PATH = os.path.join(ROOT, "results", "rl_eval.json")
TRAIN_PATH = os.path.join(ROOT, "results", "dqn_train.json")

STEPS_PER_DAY = 288  # crónicas a 5 min


def make_env():
    from grid2op.MakeEnv import make
    return make(ENV_NAME, test=True)


def build_actions(env):
    """25 acciones discretas: do-nothing + reconexiones + cambios topológicos."""
    acts = [env.action_space({})]
    acts += [env.action_space({"set_line_status": [(i, 1)]}) for i in range(env.n_line)]
    acts += [env.action_space({"change_bus": {"lines_or_id": i}}) for i in range(env.n_line)]
    acts += [env.action_space({"change_bus": {"lines_ex_id": i}}) for i in range(env.n_line)]
    return acts


ACTION_DESC = (["Do nothing"] +
               [f"Reconectar línea {i}" for i in range(8)] +
               [f"Cambiar bus origen línea {i}" for i in range(8)] +
               [f"Cambiar bus extremo línea {i}" for i in range(8)])


def legal_mask(env, obs, actions) -> np.ndarray:
    """Máscara de acciones no-ambiguas según el estado de las líneas.

    Índice 0: do-nothing (siempre legal).
    1..n_line: reconectar línea i → legal solo si está DESCONECTADA.
    1+n_line..: change_bus (origen/extremo) línea i → legal solo si está CONECTADA.
    Esto evita acciones ambiguas (que terminan el episodio) sin llamadas costosas.
    """
    nl = env.n_line
    ls = np.asarray(obs.line_status, dtype=bool)
    mask = np.zeros(len(actions), dtype=bool)
    mask[0] = True
    mask[1:1 + nl] = ~ls               # reconectar solo líneas caídas
    mask[1 + nl:1 + 2 * nl] = ls       # change_bus origen solo líneas activas
    mask[1 + 2 * nl:1 + 3 * nl] = ls   # change_bus extremo solo líneas activas
    return mask


def obs_features(obs) -> np.ndarray:
    """Vector de estado de 60 dimensiones, normalizado a rangos estables."""
    step = obs.current_step % STEPS_PER_DAY
    maint = np.clip(np.asarray(obs.time_next_maintenance, dtype=float), -1, STEPS_PER_DAY) / STEPS_PER_DAY
    cool = np.clip(np.asarray(obs.time_before_cooldown_line, dtype=float), 0, 12) / 12.0
    feats = np.concatenate([
        np.asarray(obs.rho, dtype=float),
        np.asarray(obs.line_status, dtype=float),
        np.asarray(obs.topo_vect, dtype=float) / 2.0,
        np.asarray(obs.load_p, dtype=float) / 100.0,
        np.asarray(obs.prod_p, dtype=float) / 200.0,
        maint,
        cool,
        [np.sin(2 * np.pi * step / STEPS_PER_DAY), np.cos(2 * np.pi * step / STEPS_PER_DAY)],
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)


N_FEATURES = 60
N_ACTIONS = 25


N_CHRONICS = 20  # crónicas 00..19 incluidas en rte_case5_example


def select_chronic(env, chronic_id: str):
    """Selecciona determinísticamente una crónica por su id ('00'..'19')."""
    import os
    mf = env.chronics_handler.real_data
    avail = mf.available_chronics()
    ids = [os.path.basename(a) for a in avail]
    idx = ids.index(str(chronic_id)) if str(chronic_id) in ids else 0
    probs = np.zeros(len(avail))
    probs[idx] = 1.0
    mf.sample_next_chronics(probs)
    return ids[idx]


def city_rl_config(city: str) -> dict:
    """Configuración RL por ciudad: crónica, semilla y etiqueta de severidad."""
    c = sd.get_city(city)
    import hashlib
    h = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
    chronic = f"{h % N_CHRONICS:02d}"
    severity = round((c["peak_factor"] - 1.0) * 100 + c["growth_pct"] * 8 + c["saidi_h"], 1)
    label = "Alta" if severity > 60 else ("Media" if severity > 35 else "Moderada")
    return dict(chronic=chronic, seed=(h % 10_000), severity=severity, label=label,
                n_chronics=N_CHRONICS)


def run_episode(env, agent_kind: str, actions, chronic: str, seed: int,
                max_steps: int = 600, dqn: NumpyDQN | None = None,
                record: bool = False, epsilon: float = 0.0) -> dict:
    """Ejecuta un episodio y devuelve métricas (y curva opcional)."""
    select_chronic(env, chronic)
    env.seed(seed)
    obs = env.reset()

    if agent_kind == "heuristica":
        from grid2op.Agent import PowerLineSwitch as PowerflowSwitch
        agent = PowerflowSwitch(env.action_space)

    curve = [] if record else None
    total_reward, steps_alive, rho_max_hist, overload_steps = 0.0, 0, 0.0, 0
    last_reward, done = 0.0, False

    for _ in range(max_steps):
        if agent_kind == "donothing":
            act = actions[0]
        elif agent_kind == "heuristica":
            act = agent.act(obs, last_reward, False)
        elif agent_kind == "dqn":
            s = obs_features(obs)
            mask = legal_mask(env, obs, actions)
            ai = dqn.act(s, mask, greedy=(epsilon <= 0)) if epsilon <= 0 else dqn.act(s, mask)
            act = actions[ai]
        else:
            raise ValueError(agent_kind)
        obs, reward, done, info = env.step(act)
        last_reward = float(reward)
        total_reward += float(reward)
        steps_alive += 1
        rmax = float(np.max(obs.rho)) if obs.rho is not None else 0.0
        rho_max_hist = max(rho_max_hist, rmax)
        overload_steps += int(np.any(np.asarray(obs.rho) > 1.0))
        if record:
            curve.append(dict(t=steps_alive, reward_cum=total_reward,
                              rho_max=rmax, lines_off=int((~np.asarray(obs.line_status, dtype=bool)).sum())))
        if done:
            break

    out = dict(agent=agent_kind, chronic=chronic, seed=seed,
               steps_alive=steps_alive, reward_total=round(total_reward, 2),
               rho_max=round(rho_max_hist, 3), overload_steps=overload_steps,
               survived_all=not done)
    if record:
        out["curve"] = curve
    return out


# ---------------------------------------------------------------------------
# Entrenamiento offline (scripts/train_dqn.py)
# ---------------------------------------------------------------------------
def train(episodes: int = 120, max_steps: int = 500, lr: float = 1.2e-3,
          seed: int = 7, verbose_every: int = 10, eps_decay: int = 4000) -> dict:
    env = make_env()
    actions = build_actions(env)
    chronics = [f"{i:02d}" for i in range(N_CHRONICS)]
    dqn = NumpyDQN(N_FEATURES, N_ACTIONS, lr=lr, seed=seed, eps_decay_steps=eps_decay)
    rng = np.random.default_rng(seed)
    history = []

    for ep in range(episodes):
        chronic = chronics[rng.integers(len(chronics))]
        select_chronic(env, chronic)
        env.seed(int(rng.integers(10_000)))
        obs = env.reset()
        ep_reward, ep_steps, done = 0.0, 0, False
        losses = []
        for _ in range(max_steps):
            s = obs_features(obs)
            mask = legal_mask(env, obs, actions)
            a = dqn.act(s, mask)
            obs2, r, done, _ = env.step(actions[a])
            s2 = obs_features(obs2) if not done else np.zeros(N_FEATURES, dtype=np.float32)
            mask2 = legal_mask(env, obs2, actions) if not done else np.ones(N_ACTIONS, dtype=bool)
            dqn.push(s, a, float(r), s2, done, mask2.astype(np.float32))
            loss = dqn.train_step()
            if loss is not None:
                losses.append(loss)
            ep_reward += float(r)
            ep_steps += 1
            obs = obs2
            if done:
                break
        history.append(dict(episode=ep + 1, chronic=chronic, steps_alive=ep_steps,
                            reward=round(ep_reward, 1), epsilon=round(dqn.epsilon(), 3),
                            loss=round(float(np.mean(losses)), 4) if losses else None))
        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f"ep {ep+1:3d} | chronic {chronic} | alive {ep_steps:4d} | "
                  f"R {ep_reward:8.1f} | eps {dqn.epsilon():.2f} | "
                  f"loss {history[-1]['loss']}", flush=True)

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    dqn.save(WEIGHTS_PATH)
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        json.dump(history, f)
    return dict(history=history, weights=WEIGHTS_PATH)


def load_agent() -> NumpyDQN | None:
    if os.path.exists(WEIGHTS_PATH):
        return NumpyDQN.load(WEIGHTS_PATH)
    return None


# ---------------------------------------------------------------------------
# Evaluación offline (scripts/eval_rl_agents.py → results/rl_eval.json)
# ---------------------------------------------------------------------------
def evaluate(max_steps: int = 800, agents=("donothing", "heuristica", "dqn"),
           chronics_list=None) -> dict:
    env = make_env()
    actions = build_actions(env)
    chronics = [f"{i:02d}" for i in range(N_CHRONICS)] if chronics_list is None else list(chronics_list)
    dqn = load_agent()
    rows, curves = [], {}
    for ag in agents:
        if ag == "dqn" and dqn is None:
            continue
        for ch in chronics:
            rec = (ag in agents and ch == chronics[0])
            r = run_episode(env, ag, actions, ch, seed=4242, max_steps=max_steps,
                            dqn=dqn, record=rec)
            rows.append({k: v for k, v in r.items() if k != "curve"})
            if rec:
                curves[ag] = r["curve"]
        print(f"agente {ag}: {len(rows)} episodios acumulados", flush=True)
    out = dict(rows=rows, curves=curves, max_steps=max_steps, env=ENV_NAME,
               n_chronics=len(chronics))
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(out, f)
    return out


def load_eval() -> dict | None:
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Utilidades para el dashboard (evaluación en vivo e inspección de política)
# ---------------------------------------------------------------------------
def live_eval(city: str, agents: tuple = ("donothing", "dqn"), max_steps: int = 400,
              seed: int | None = None) -> list[dict]:
    """Evaluación en vivo para la ciudad seleccionada (config crónica+semilla)."""
    env = make_env()
    actions = build_actions(env)
    cfg = city_rl_config(city)
    dqn = load_agent()
    s = cfg["seed"] if seed is None else seed
    rows = []
    for ag in agents:
        if ag == "dqn" and dqn is None:
            continue
        r = run_episode(env, ag, actions, chronic=cfg["chronic"], seed=s,
                        max_steps=max_steps, dqn=dqn, record=True)
        r["city"] = city
        rows.append(r)
    return rows


def load_train_history() -> list | None:
    if os.path.exists(TRAIN_PATH):
        with open(TRAIN_PATH) as f:
            return json.load(f)
    return None


def inspect_policy(city: str, max_steps: int = 300) -> dict | None:
    """Avanza con do-nothing hasta congestión (rho>1) y muestra los Q-values del
    DQN en ese estado crítico: evidencia de qué acción correctiva aprendió."""
    dqn = load_agent()
    if dqn is None:
        return None
    env = make_env()
    actions = build_actions(env)
    cfg = city_rl_config(city)
    select_chronic(env, cfg["chronic"])
    env.seed(cfg["seed"])
    obs = env.reset()
    for _ in range(max_steps):
        if float(np.max(obs.rho)) > 1.0:
            break
        obs, r, done, _ = env.step(actions[0])
        if done:
            break
    s = obs_features(obs)
    mask = legal_mask(env, obs, actions)
    q = dqn.q_values(s)[0]
    qm = np.where(mask, q, -np.inf)
    order = np.argsort(-qm)[:8]
    top = [dict(accion=ACTION_DESC[i], q=float(q[i]), legal=bool(mask[i]))
           for i in order if np.isfinite(qm[i])]
    return dict(step=int(obs.current_step), chronic=cfg["chronic"],
                rho=np.asarray(obs.rho).round(2).tolist(),
                rho_max=round(float(np.max(obs.rho)), 2), top=top)


# ---------------------------------------------------------------------------
# Grid2Op POR CIUDAD (ambientes con crónicas derivadas de la demanda sintética)
# ---------------------------------------------------------------------------
CITY_EVAL_PATH = os.path.join(ROOT, "results", "rl_city_eval.json")


def load_city_eval() -> dict | None:
    """Carga la evaluación precomputada por ciudad (results/rl_city_eval.json)."""
    if os.path.exists(CITY_EVAL_PATH):
        with open(CITY_EVAL_PATH) as f:
            return json.load(f)
    return None


def live_eval_city(city: str, agents=("donothing", "dqn"), max_steps: int = 300,
                   chronic: str = "00") -> list[dict]:
    """Corre episodios EN VIVO en el ambiente Grid2Op propio de la ciudad
    (crónica re-formada desde su curva de demanda). Devuelve métricas + curva."""
    import core.city_env as ce
    env = ce.make_city_env(city)
    try:
        actions = build_actions(env)
        dqn = load_agent()
        seed = city_rl_config(city)["seed"]
        rows = []
        for a in agents:
            r = run_episode(env, a, actions, chronic, seed=seed,
                            max_steps=max_steps, dqn=dqn, record=True)
            rows.append(r)
        return rows
    finally:
        env.close()

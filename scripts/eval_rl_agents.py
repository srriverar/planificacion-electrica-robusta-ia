# -*- coding: utf-8 -*-
"""Evaluación offline de agentes RL sobre Grid2Op (rte_case5_example).

Compara Do-Nothing, heurística PowerLineSwitch y el DQN pre-entrenado en las
20 crónicas semanales. Guarda results/rl_eval.json para que el dashboard
muestre resultados instantáneos (y ofrezca evaluación en vivo opcional).

Uso:  python scripts/eval_rl_agents.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from core.grid2op_rl import (EVAL_PATH, N_CHRONICS, build_actions, load_agent,  # noqa: E402
                             make_env, run_episode)

MAX_STEPS_FAST = 800     # donothing / dqn
MAX_STEPS_HEUR = 400     # heurística (más lenta por paso: power flow por acción)
HEUR_CHRONICS = ["00", "04", "09", "13", "17"]
CURVE_CHRONIC = "00"     # crónica donde do-nothing muere temprano (buen contraste)
SEED = 4242


def main():
    env = make_env()
    actions = build_actions(env)
    dqn = load_agent()
    if dqn is None:
        raise SystemExit("No hay pesos entrenados. Ejecuta primero scripts/train_dqn.py")
    chronics = [f"{i:02d}" for i in range(N_CHRONICS)]
    rows, curves = [], {}

    plan = [("donothing", chronics, MAX_STEPS_FAST),
            ("dqn", chronics, MAX_STEPS_FAST),
            ("heuristica", HEUR_CHRONICS, MAX_STEPS_HEUR)]
    t0 = time.time()
    for ag, chs, max_steps in plan:
        for ch in chs:
            rec = (ch == CURVE_CHRONIC)
            r = run_episode(env, ag, actions, chronic=ch, seed=SEED,
                            max_steps=max_steps, dqn=dqn, record=rec)
            rows.append({k: v for k, v in r.items() if k != "curve"})
            if rec:
                curves[ag] = r["curve"]
            print(f"[{time.time()-t0:6.1f}s] {ag:11s} ch={ch} alive={r['steps_alive']:4d} "
                  f"R={r['reward_total']:8.1f} rho_max={r['rho_max']:.2f}", flush=True)

    out = dict(rows=rows, curves=curves, max_steps=MAX_STEPS_FAST,
               max_steps_heur=MAX_STEPS_HEUR, env="rte_case5_example",
               n_chronics=N_CHRONICS, seed=SEED)
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(out, f)
    print("Guardado:", EVAL_PATH, flush=True)


if __name__ == "__main__":
    main()

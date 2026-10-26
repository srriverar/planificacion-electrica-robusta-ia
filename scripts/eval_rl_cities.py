"""Evalúa el agente DQN entrenado vs Do-Nothing en el ambiente Grid2Op **de cada
ciudad** (crónicas re-formadas desde la demanda sintética de la ciudad).

Guarda results/rl_city_eval.json con, por ciudad: métricas agregadas de ambos agentes
sobre las 5 crónicas-semana y las curvas (reward acumulado y rho_max) de la crónica 00.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.city_env as ce
import core.grid2op_rl as g
import core.synthetic_data as sd

MAX_STEPS = 600
SEED = 7
OUT = os.path.join(ROOT, "results", "rl_city_eval.json")


def _subsample(curve, every=10):
    return [c for i, c in enumerate(curve) if i % every == 0 or i == len(curve) - 1]


def main():
    cities = sd.list_cities()
    dqn = g.load_agent()
    assert dqn is not None, "Faltan los pesos del DQN (models/dqn_case5.npz)"

    out = {"env": "rte_case5_example + crónicas por ciudad (city_env)",
           "max_steps": MAX_STEPS, "seed": SEED, "n_chronics": ce.N_CHRONICS_CITY,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "cities": {}}

    t00 = time.time()
    for i, city in enumerate(cities, 1):
        t0 = time.time()
        env = ce.make_city_env(city)          # cacheado en disco
        actions = g.build_actions(env)
        summ = ce.city_env_summary(city)
        chron_ids = [f"{v:02d}" for v in range(ce.N_CHRONICS_CITY)]

        agg = {}
        curves = {}
        for agent in ("donothing", "dqn"):
            steps, rewards, rhos, survived = [], [], [], 0
            for cid in chron_ids:
                rec = (cid == "00")
                res = g.run_episode(env, agent, actions, cid, seed=SEED,
                                    max_steps=MAX_STEPS, dqn=dqn, record=rec)
                steps.append(res["steps_alive"])
                rewards.append(res["reward_total"])
                rhos.append(res["rho_max"])
                survived += int(res["survived_all"])
                if rec:
                    curves[agent] = _subsample(res["curve"], 10)
            agg[agent] = {
                "steps": steps, "steps_mean": round(sum(steps) / len(steps), 1),
                "steps_total": int(sum(steps)),
                "reward_total": round(sum(rewards), 1),
                "reward_mean": round(sum(rewards) / len(rewards), 1),
                "rho_max": round(max(rhos), 3),
                "survived": survived, "n": len(steps),
            }

        ventaja = round(agg["dqn"]["steps_total"] / max(agg["donothing"]["steps_total"], 1), 2)
        out["cities"][city] = {
            "summary": summ, "chronic_ids": chron_ids,
            "donothing": agg["donothing"], "dqn": agg["dqn"],
            "ventaja": ventaja, "curves": curves,
        }
        env.close()
        print(f"[{i:2d}/{len(cities)}] {city:15s} peak={summ['peak_load_mw']:5.1f}MW "
              f"ren={summ['ren_share']:.2f} | DN {agg['donothing']['steps_mean']:5.1f}p "
              f"({agg['donothing']['survived']}/{len(steps)}) | "
              f"DQN {agg['dqn']['steps_mean']:5.1f}p ({agg['dqn']['survived']}/{len(steps)}) "
              f"| x{ventaja} | {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"\nGuardado: {OUT}  ({time.time()-t00:.0f}s totales)", flush=True)


if __name__ == "__main__":
    main()

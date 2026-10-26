# -*- coding: utf-8 -*-
"""Entrenamiento offline del DQN (Grid2Op rte_case5_example).

Uso:  python scripts/train_dqn.py [episodes]
Genera: models/dqn_case5.npz y results/dqn_train.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid2op_rl import train  # noqa: E402

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    print(f"Entrenando DQN: {episodes} episodios × máx {max_steps} pasos", flush=True)
    out = train(episodes=episodes, max_steps=max_steps, verbose_every=5)
    print("Pesos guardados en:", out["weights"])

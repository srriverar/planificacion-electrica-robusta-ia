# -*- coding: utf-8 -*-
"""
DQN minimalista implementado en NumPy puro (sin dependencias pesadas), para el
agente de aprendizaje por refuerzo que opera la red Grid2Op.

Arquitectura: MLP 60 → 64 → 64 → 25 (ReLU), Adam, replay buffer, target network,
ε-greedy con decaimiento y máscara de acciones legales.

Este módulo es intencionalmente compacto y reproducible: los pesos entrenados
offline se guardan en models/dqn_case5.npz y el dashboard los carga en segundos.
"""
from __future__ import annotations

import numpy as np


def _he_init(rng, fan_in, fan_out):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class NumpyDQN:
    def __init__(self, n_features: int, n_actions: int, hidden=(64, 64),
                 lr: float = 1e-3, gamma: float = 0.99, buffer_size: int = 20_000,
                 batch_size: int = 64, seed: int = 0, target_sync: int = 400,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 6000):
        self.n_features, self.n_actions = n_features, n_actions
        self.lr, self.gamma = lr, gamma
        self.buffer_size, self.batch_size = buffer_size, batch_size
        self.target_sync = target_sync
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        self.params = {
            "W1": _he_init(rng, n_features, h1), "b1": np.zeros(h1),
            "W2": _he_init(rng, h1, h2), "b2": np.zeros(h2),
            "W3": _he_init(rng, h2, n_actions), "b3": np.zeros(n_actions),
        }
        self.target = {k: v.copy() for k, v in self.params.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self._eps = eps_start
        # replay buffer
        self.buf_s = np.zeros((buffer_size, n_features), dtype=np.float32)
        self.buf_s2 = np.zeros((buffer_size, n_features), dtype=np.float32)
        self.buf_m2 = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.buf_a = np.zeros(buffer_size, dtype=np.int64)
        self.buf_r = np.zeros(buffer_size, dtype=np.float32)
        self.buf_d = np.zeros(buffer_size, dtype=np.float32)
        self.buf_n = 0
        self.buf_i = 0

    # ------------------------------------------------------------------
    def _forward(self, p, X):
        z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        q = a2 @ p["W3"] + p["b3"]
        return q, (X, z1, a1, z2, a2)

    def q_values(self, X, use_target=False):
        p = self.target if use_target else self.params
        q, _ = self._forward(p, np.atleast_2d(X).astype(np.float32))
        return q

    def epsilon(self):
        frac = min(1.0, self.t / max(self.eps_decay_steps, 1))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, s, legal_mask, greedy=False, explore_donothing_p: float = 0.75):
        """Acción ε-greedy con máscara de legalidad.

        En exploración, con probabilidad `explore_donothing_p` se hace do-nothing
        (índice 0): en Grid2Op las acciones aleatorias suelen ser fatales
        (aíslan nodos), por lo que este sesgo alarga los episodios y permite
        aprender correcciones ante hazards/mantenimientos (práctica L2RPN).
        """
        if not greedy and np.random.rand() < self.epsilon():
            if legal_mask[0] and np.random.rand() < explore_donothing_p:
                return 0
            legal_idx = np.flatnonzero(legal_mask)
            if len(legal_idx) == 0:
                return 0
            return int(legal_idx[np.random.randint(len(legal_idx))])
        q = self.q_values(s)[0]
        q = np.where(legal_mask, q, -np.inf)
        if not np.isfinite(q).any():
            return 0
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    def push(self, s, a, r, s2, done, legal_mask2):
        i = self.buf_i
        self.buf_s[i] = s; self.buf_s2[i] = s2; self.buf_m2[i] = legal_mask2
        self.buf_a[i] = a; self.buf_r[i] = r; self.buf_d[i] = float(done)
        self.buf_i = (i + 1) % self.buffer_size
        self.buf_n = min(self.buf_n + 1, self.buffer_size)

    def train_step(self):
        if self.buf_n < max(self.batch_size * 8, 512):
            return None
        idx = np.random.randint(0, self.buf_n, size=self.batch_size)
        s = self.buf_s[idx]; a = self.buf_a[idx]; r = self.buf_r[idx]
        s2 = self.buf_s2[idx]; d = self.buf_d[idx]; m2 = self.buf_m2[idx]

        q2, _ = self._forward(self.target, s2)
        q2 = np.where(m2 > 0, q2, -np.inf)
        q2max = np.where(np.isfinite(q2).any(axis=1), q2.max(axis=1), 0.0)
        y = r + self.gamma * (1.0 - d) * q2max

        q, cache = self._forward(self.params, s)
        X, z1, a1, z2, a2 = cache
        q_sel = q[np.arange(self.batch_size), a]
        err = q_sel - y                                   # (B,)
        # Huber
        huber_grad = np.clip(err, -1.0, 1.0) / self.batch_size
        dq = np.zeros_like(q)
        dq[np.arange(self.batch_size), a] = huber_grad

        gW3 = a2.T @ dq
        gb3 = dq.sum(axis=0)
        da2 = dq @ self.params["W3"].T
        dz2 = da2 * (z2 > 0)
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        gW1 = X.T @ dz1
        gb1 = dz1.sum(axis=0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}

        self.t += 1
        b1_, b2_, eps_ = 0.9, 0.999, 1e-8
        for k in self.params:
            self.m[k] = b1_ * self.m[k] + (1 - b1_) * grads[k]
            self.v[k] = b2_ * self.v[k] + (1 - b2_) * grads[k] ** 2
            mh = self.m[k] / (1 - b1_ ** self.t)
            vh = self.v[k] / (1 - b2_ ** self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + eps_)
        if self.t % self.target_sync == 0:
            self.target = {k: v.copy() for k, v in self.params.items()}
        return float(np.mean(np.abs(err)))

    # ------------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in self.params.items()},
                            n_features=self.n_features, n_actions=self.n_actions)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obj = cls(int(z["n_features"]), int(z["n_actions"]))
        for k in obj.params:
            obj.params[k] = z[k].astype(np.float64)
        obj.target = {k: v.copy() for k, v in obj.params.items()}
        return obj
